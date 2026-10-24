"""Training loss and the logit-derived helpers used at evaluation time."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """Cross-entropy over the 4 answer-letter logits.

    Equivalent to SFT on the single answer token with the softmax restricted
    to {A, B, C, D}; the other ~250k vocabulary entries get no gradient, so
    the model is never pushed to rank a letter over a non-letter.
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce(logits.float(), labels)


def unlikelihood(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean over rows of -log(1 - p(label)), p = softmax over the 4 letters.

    Used on the counterfactual (time-reversed) view: it pushes probability
    away from the original answer without naming a new one, and is ~0 once
    that answer is unlikely. log(1 - p_y) = logsumexp(others) - logsumexp(all).
    """
    logits = logits.float()
    others = logits.masked_fill(
        F.one_hot(labels, logits.shape[-1]).bool(), float("-inf")
    )
    return (logits.logsumexp(-1) - others.logsumexp(-1)).mean()


@torch.no_grad()
def probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Softmax probabilities, (B, num_classes)."""
    return F.softmax(logits.float(), dim=1)


@torch.no_grad()
def predict(logits: torch.Tensor) -> torch.Tensor:
    """Argmax class predictions, (B,)."""
    return logits.argmax(dim=1)
