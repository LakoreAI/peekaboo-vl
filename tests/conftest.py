import pytest

from src.data import ReVAExample


def make_example(
    i: int = 0,
    video: str = "VisDrone/uav0000009_03358_v_01.mp4",
    label: int = 1,
    task: str = "Temporal Grounding",
    question: str = None,
) -> ReVAExample:
    return ReVAExample(
        qa_id=f"train_{i:06d}",
        video_path=video,
        task=task,
        category="Temporal Understanding",
        question=question or f"When does object {i} appear?",
        options=(f"opt{i}-a", f"opt{i}-b", f"opt{i}-c", f"opt{i}-d"),
        label=label,
        dataset_name="VisDrone",
        subdir="VisDrone",
    )


@pytest.fixture
def example():
    return make_example()


@pytest.fixture
def official_record():
    return {
        "qa_id": "val_000001",
        "video_path": "Hawk_UAV/NJ/DJI_0157_d4_01.mp4",
        "subdir": "NJ",
        "dataset_name": "Hawk_UAV",
        "category": "Temporal Understanding",
        "task": "Temporal Grounding",
        "question": "At what point in the video does the road with moving vehicles first appear?",
        "options": {
            "A": "At 5.0 seconds",
            "B": "At 4.2 seconds",
            "C": "At 2.5 seconds",
            "D": "At 3.8 seconds",
        },
        "correct_answer": "B",
    }
