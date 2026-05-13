import pytest
from modules.audio_features import AudioRiskEstimator


class TestAudioRiskEstimator:
    def setup_method(self):
        self.estimator = AudioRiskEstimator()

    def test_score_within_bounds(self):
        features = {"rms": 0.05, "long_silences_ratio": 0.3}
        score = self.estimator.estimate_risk(features)
        assert 0.0 <= score <= 1.0

    def test_high_silence_raises_risk(self):
        low_silence = self.estimator.estimate_risk({"rms": 0.1, "long_silences_ratio": 0.1})
        high_silence = self.estimator.estimate_risk({"rms": 0.1, "long_silences_ratio": 0.9})
        assert high_silence > low_silence

    def test_missing_keys_default_to_zero(self):
        score = self.estimator.estimate_risk({})
        assert score == pytest.approx(0.0, abs=0.16)

    def test_max_score_clamps_to_one(self):
        score = self.estimator.estimate_risk({"rms": 0.0, "long_silences_ratio": 1.0})
        assert score <= 1.0
