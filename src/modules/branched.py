"""Shared-prefix ("branched") scoring: one video, k questions, no leakage.

Packing put k questions in one sequence, so later questions saw the
earlier gold answers. Here the shared prefix (system + video + any common
text) runs once, and each question's suffix branches off it:

    prefix (1, P) ──────────────► state after P tokens
                                   ├─► suffix 1 (Q1 + gen prompt) -> letters
                                   ├─► suffix 2 ...
                                   └─► suffix k

- Full-attention layers: each suffix attends to the prefix keys/values and
  causally to its own tokens.
- Gated DeltaNet layers: each suffix starts from the prefix's recurrent state
  and the last `kernel-1` conv inputs.

Every suffix therefore sees exactly the tokens it would see as a standalone
single-question prompt (the same test-time input), and gradients from all k
suffixes add up through the shared prefix. Suffixes are right-padded: with
causal mixing, pads after the last real token never reach it.

Qwen3.5 only (hybrid Gated DeltaNet + gated full attention); the decoder
layers are driven here with their own submodules, so LoRA on any of their
linears is used as usual.
"""

import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _parts(backbone):
    """(Qwen3_5Model, text model, lm_head, modeling module) under PEFT."""
    base = (
        backbone.get_base_model() if hasattr(backbone, "get_base_model") else backbone
    )
    mm = base.model
    lm = mm.language_model
    mod = sys.modules[type(lm).__module__]
    if not hasattr(mod, "torch_chunk_gated_delta_rule"):
        raise NotImplementedError(
            f"branched scoring supports Qwen3.5 only, got {mod.__name__}"
        )
    return mm, lm, base.lm_head, mod


# --- per-layer token mixers ---------------------------------------------------


def _gdn(M, m, x, conv_tail=None, init_state=None):
    """Gated DeltaNet on x (B, L, d), optionally continuing from a prefix.

    Mirrors `Qwen3_5GatedDeltaNet.forward` (no cache, no padding mask).
    Returns (output, last kernel-1 conv inputs, final recurrent state)."""
    B, L, _ = x.shape
    mixed = m.in_proj_qkv(x).transpose(1, 2)
    z = m.in_proj_z(x).reshape(B, L, -1, m.head_v_dim)
    b = m.in_proj_b(x)
    a = m.in_proj_a(x)

    K = m.conv_kernel_size
    tail = mixed[:, :, -(K - 1) :]
    if conv_tail is not None:
        mixed = torch.cat([conv_tail.expand(B, -1, -1), mixed], dim=-1)
    conv = M.causal_conv1d_fn(
        mixed.contiguous(),
        m.conv1d.weight.squeeze(1),
        m.conv1d.bias,
        activation=m.activation,
    )
    conv = conv[:, :, -L:].transpose(1, 2)
    q, k, v = torch.split(conv, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
    q = q.reshape(B, L, -1, m.head_k_dim)
    k = k.reshape(B, L, -1, m.head_k_dim)
    v = v.reshape(B, L, -1, m.head_v_dim)
    beta = b.sigmoid()
    g = -m.A_log.float().exp() * F.softplus(a.float() + m.dt_bias)
    rep = m.num_v_heads // m.num_k_heads
    if rep > 1:
        q = q.repeat_interleave(rep, dim=2)
        k = k.repeat_interleave(rep, dim=2)
    if init_state is not None and init_state.shape[0] != B:
        init_state = init_state.expand(B, *init_state.shape[1:]).contiguous()
    out, state = M.torch_chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=init_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    out = m.norm(out.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim))
    return m.out_proj(out.reshape(B, L, -1)), tail, state


def _qkv(M, m, x, cos, sin):
    B, L, _ = x.shape
    shape = (B, L, -1, m.head_dim)
    q, gate = torch.chunk(m.q_proj(x).view(B, L, -1, m.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(B, L, -1)
    q = m.q_norm(q.view(shape)).transpose(1, 2)
    k = m.k_norm(m.k_proj(x).view(shape)).transpose(1, 2)
    v = m.v_proj(x).view(shape).transpose(1, 2)
    q, k = M.apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, v, gate


def _attn_out(m, o, gate):
    B, _, L, _ = o.shape
    o = o.transpose(1, 2).reshape(B, L, -1) * torch.sigmoid(gate)
    return m.o_proj(o)


def _attention(M, m, xp, xss, pe_p, pe_ss):
    """Gated full attention: prefix causal; each suffix group sees the prefix
    and its own past."""
    qp, kp, vp, gp = _qkv(M, m, xp, *pe_p)
    n = m.num_key_value_groups
    op = F.scaled_dot_product_attention(
        qp, M.repeat_kv(kp, n), M.repeat_kv(vp, n), is_causal=True, scale=m.scaling
    )
    outs = []
    for xs, pe_s in zip(xss, pe_ss):
        qs, ks, vs, gs = _qkv(M, m, xs, *pe_s)
        B, S, P = xs.shape[0], xs.shape[1], xp.shape[1]
        kc = torch.cat([kp.expand(B, -1, -1, -1), ks], dim=2)
        vc = torch.cat([vp.expand(B, -1, -1, -1), vs], dim=2)
        mask = torch.ones(S, P + S, dtype=torch.bool, device=xs.device)
        mask[:, P:] = torch.ones(S, S, dtype=torch.bool, device=xs.device).tril()
        os_ = F.scaled_dot_product_attention(
            qs, M.repeat_kv(kc, n), M.repeat_kv(vc, n), attn_mask=mask, scale=m.scaling
        )
        outs.append(_attn_out(m, os_, gs))
    return _attn_out(m, op, gp), outs


def _layer(M, layer, hp, hss, pe_p, pe_ss):
    xp = layer.input_layernorm(hp)
    xss = [layer.input_layernorm(hs) for hs in hss]
    if layer.block_type == "linear_attention":
        op, tail, state = _gdn(M, layer.linear_attn, xp)
        oss = [
            _gdn(M, layer.linear_attn, xs, conv_tail=tail, init_state=state)[0]
            for xs in xss
        ]
    else:
        op, oss = _attention(M, layer.self_attn, xp, xss, pe_p, pe_ss)
    hp = hp + op
    hp = hp + layer.mlp(layer.post_attention_layernorm(hp))
    out = []
    for hs, os_ in zip(hss, oss):
        hs = hs + os_
        out.append(hs + layer.mlp(layer.post_attention_layernorm(hs)))
    return hp, out


# --- full forward -------------------------------------------------------------


def branched_forward(
    backbone,
    inputs: dict,
    prefix_len: int,
    groups: List[List[int]],
    grad_ckpt: bool,
) -> List[torch.Tensor]:
    """Final hidden states of the suffixes, one tensor per row group.

    inputs: right-padded rows (R, L) (`input_ids`, `attention_mask`,
      `mm_token_type_ids`) and the video tensors for ONE copy
      (`pixel_values_videos`, `video_grid_thw` of shape (1, 3)).
    prefix_len: P, the token prefix shared by all rows; it must contain every
      video token.
    groups: row indices per suffix group, e.g. [[0..k-1], [k]] for k question
      rows plus one long fact row. Each group is padded only to its own longest
      row. Returns, per group, hidden states (len(group), S_g, d) for absolute
      positions P .. P + S_g - 1.
    """
    mm, lm, _, M = _parts(backbone)
    ids = inputs["input_ids"]
    R, P = ids.shape[0], prefix_len
    grid = inputs["video_grid_thw"]
    lengths = inputs["attention_mask"].sum(-1)

    pos, _ = mm.get_rope_index(
        ids,
        mm_token_type_ids=inputs["mm_token_type_ids"],
        video_grid_thw=grid.repeat(R, 1),
        attention_mask=inputs["attention_mask"],
    )
    embed = mm.get_input_embeddings()
    hp = embed(ids[:1, :P])
    feats = mm.get_video_features(inputs["pixel_values_videos"], grid, return_dict=True)
    video = torch.cat(feats.pooler_output, dim=0).to(hp.device, hp.dtype)
    vmask = (ids[:1, :P] == mm.config.video_token_id).unsqueeze(-1)
    if int(vmask.sum()) != video.shape[0]:
        raise ValueError("video tokens are not all inside the shared prefix")
    hp = hp.masked_scatter(vmask.expand_as(hp), video)

    hss, pe_ss = [], []
    for g in groups:
        idx = torch.as_tensor(g, device=ids.device)
        end = int(lengths[idx].max())
        hs = embed(ids[idx, P:end])
        hss.append(hs)
        pe_ss.append(lm.rotary_emb(hs, pos[:, idx, P:end]))
    pe_p = lm.rotary_emb(hp, pos[:, :1, :P])
    for layer in lm.layers[: lm.config.num_hidden_layers]:
        args = (M, layer, hp, hss, pe_p, pe_ss)
        if grad_ckpt and torch.is_grad_enabled():
            hp, hss = checkpoint(_layer, *args, use_reentrant=False)
        else:
            hp, hss = _layer(*args)
    return [lm.norm(hs) for hs in hss]


def branched_hidden(
    backbone, inputs: dict, prefix_len: int, last_index: torch.Tensor, grad_ckpt: bool
) -> torch.Tensor:
    """Final hidden state (k, d) at each row's answer position (all rows are
    single-question prompts; `last_index` = each row's last real token)."""
    k = inputs["input_ids"].shape[0]
    (hs,) = branched_forward(backbone, inputs, prefix_len, [list(range(k))], grad_ckpt)
    rows = torch.arange(k, device=hs.device)
    return hs[rows, last_index - prefix_len]


def shared_prefix_len(rows: List[List[int]]) -> int:
    """Longest common prefix of token rows, leaving >= 1 suffix token each."""
    n = min(len(r) for r in rows) - 1
    i = 0
    while i < n and all(r[i] == rows[0][i] for r in rows[1:]):
        i += 1
    return i


def right_pad(rows: List[List[int]], pad: int) -> Tuple[torch.Tensor, torch.Tensor]:
    L = max(len(r) for r in rows)
    ids = torch.full((len(rows), L), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), L), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.as_tensor(r)
        mask[i, : len(r)] = 1
    return ids, mask
