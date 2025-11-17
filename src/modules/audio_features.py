from typing import Dict

from utils.audio_utils import extract_basic_features


class AudioRiskEstimator:
    def __init__(self) -> None:
        pass

    def estimate_risk(self, features: Dict[str, float]) -> float:
        rms = features.get("rms", 0.0)
        silences = features.get("long_silences_ratio", 0.0)
        base = 0.5 * silences + 0.5 * max(0.0, 0.3 - rms)
        score = max(0.0, min(1.0, base))
        return score
