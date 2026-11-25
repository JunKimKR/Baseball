"""MediaPipe Pose 33-landmark names and side-aware joint lookup."""

JOINTS: list[str] = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

_INDEX = {name: i for i, name in enumerate(JOINTS)}


def idx(name: str) -> int:
    """Index of a joint name in the MediaPipe ordering."""
    return _INDEX[name]


def side_joints(hand: str) -> dict[str, int]:
    """Role -> joint index for a pitcher's throwing hand ("R" or "L").

    The throwing side is the hand's side; the lead (stride) leg is the opposite side.
    """
    if hand not in ("R", "L"):
        raise ValueError(f"hand must be 'R' or 'L', got {hand!r}")
    throw, lead = ("right", "left") if hand == "R" else ("left", "right")
    return {
        "throw_shoulder": idx(f"{throw}_shoulder"),
        "throw_elbow": idx(f"{throw}_elbow"),
        "throw_wrist": idx(f"{throw}_wrist"),
        "lead_hip": idx(f"{lead}_hip"),
        "lead_knee": idx(f"{lead}_knee"),
        "lead_ankle": idx(f"{lead}_ankle"),
        "lead_heel": idx(f"{lead}_heel"),
        "lead_toe": idx(f"{lead}_foot_index"),
        "rear_hip": idx(f"{throw}_hip"),
        "rear_knee": idx(f"{throw}_knee"),
        "rear_ankle": idx(f"{throw}_ankle"),
    }
