from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class RiskResult:
    text_risk: float
    audio_risk: float
    final_risk: float
    risk_label: str
    text: str = ""
    extra: Optional[Dict[str, Any]] = None


def combine_risk(text_risk: float, audio_risk: float) -> float:
    return 0.6 * text_risk + 0.4 * audio_risk


def label_from_score(score: float, high: float = 0.8, medium: float = 0.5) -> str:
    if score >= high:
        return "high"
    if score >= medium:
        return "medium"
    return "low"
