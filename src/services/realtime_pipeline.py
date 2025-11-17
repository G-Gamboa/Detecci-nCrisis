from dataclasses import dataclass
from typing import Optional

from utils.audio_utils import load_audio, extract_basic_features
from utils.risk_rules import combine_risk, label_from_score, RiskResult
from modules.speech_to_text import SpeechToTextEngine
from modules.text_classifier import TextRiskClassifier
from modules.audio_features import AudioRiskEstimator
from modules.alerting import AlertService


@dataclass
class PipelineConfig:
    high_threshold: float = 0.8
    medium_threshold: float = 0.5


class RealtimePipeline:
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.asr = SpeechToTextEngine()
        self.text_clf = TextRiskClassifier()
        self.audio_estimator = AudioRiskEstimator()
        self.alert_service = AlertService()

    def process_file(self, audio_path: str) -> RiskResult:
        text = self.asr.transcribe_file(audio_path)
        text_probs = self.text_clf.predict_proba([text])
        text_risk = text_probs[0]
        signal, sr = load_audio(audio_path)
        features = extract_basic_features(signal, sr)
        audio_risk = self.audio_estimator.estimate_risk(features)
        final_risk = combine_risk(text_risk, audio_risk)
        label = label_from_score(
            final_risk,
            high=self.config.high_threshold,
            medium=self.config.medium_threshold,
        )
        result = RiskResult(
            text_risk=text_risk,
            audio_risk=audio_risk,
            final_risk=final_risk,
            risk_label=label,
            text=text,
            extra={"audio_features": features},
        )
        if label == "high":
            self.alert_service.send_high_risk_alert(final_risk, text)
        return result
