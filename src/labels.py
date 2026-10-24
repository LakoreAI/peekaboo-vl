"""ReVA answer letters and question taxonomy.

The four answer letters are the classes: the model scores the next-token logits
of "A", "B", "C", "D" after the prompt, so a 4-way classifier sits on top of
the language-model head. The 11 tasks and 4 categories come from the
annotation files' `metadata.question_taxonomy`; the short names match the
Codabench leaderboard columns.
"""

LETTERS = ("A", "B", "C", "D")
NUM_OPTIONS = len(LETTERS)
LETTER_TO_INDEX = {letter: i for i, letter in enumerate(LETTERS)}
# Display letters for questions with more options (external benchmarks such as
# UrbanVideo-Bench have up to 7); ReVA questions only ever use the first four.
OPTION_LETTERS = ("A", "B", "C", "D", "E", "F", "G")

# task -> (category, leaderboard column)
TASKS = {
    "General Understanding": ("Factual Perception", "FP_General"),
    "Object and Land Cover Recognition": ("Factual Perception", "FP_ObjectLand"),
    "Change Detection": ("Factual Perception", "FP_Change"),
    "Temporal Grounding": ("Temporal Understanding", "TU_Temporal"),
    "Trend and Pattern": ("Temporal Understanding", "TU_TrendPattern"),
    "Geometric Relation": ("Spatial Reasoning", "SR_Geometric"),
    "Structural Layout": ("Spatial Reasoning", "SR_Structural"),
    "Perspective and Viewpoint": ("Spatial Reasoning", "SR_Viewpoint"),
    "Causation Reasoning": ("Causal Reasoning", "CR_Causation"),
    "Consequence Reasoning": ("Causal Reasoning", "CR_Consequence"),
    "Hypothetical Reasoning": ("Causal Reasoning", "CR_Hypothetical"),
}
CATEGORIES = tuple(dict.fromkeys(cat for cat, _ in TASKS.values()))

# Question counts in the hidden test set (Codabench page / test.json
# metadata). Used to report a test-weighted accuracy on dev, since dev's task
# mix differs from test's.
TEST_TASK_COUNTS = {
    "General Understanding": 180,
    "Object and Land Cover Recognition": 660,
    "Change Detection": 500,
    "Temporal Grounding": 640,
    "Trend and Pattern": 360,
    "Geometric Relation": 400,
    "Structural Layout": 340,
    "Perspective and Viewpoint": 480,
    "Causation Reasoning": 180,
    "Consequence Reasoning": 100,
    "Hypothetical Reasoning": 160,
}


def task_short(task: str) -> str:
    """Leaderboard column for a task name; unknown tasks pass through."""
    return TASKS[task][1] if task in TASKS else task


def task_category(task: str) -> str:
    return TASKS[task][0] if task in TASKS else "Unknown"
