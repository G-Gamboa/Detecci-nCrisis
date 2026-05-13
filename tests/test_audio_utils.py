import numpy as np
import pytest
from utils.audio_utils import extract_basic_features


def _sine_wave(freq: float = 440.0, duration: float = 1.0, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silent_signal(duration: float = 1.0, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(sr * duration), dtype=np.float32)


class TestExtractBasicFeatures:
    def test_returns_expected_keys(self):
        signal = _sine_wave()
        features = extract_basic_features(signal, 16000)
        assert "rms" in features
        assert "zcr" in features
        assert "long_silences_ratio" in features

    def test_values_are_floats(self):
        signal = _sine_wave()
        features = extract_basic_features(signal, 16000)
        for key, val in features.items():
            assert isinstance(val, float), f"{key} should be float"

    def test_silence_has_high_ratio(self):
        features = extract_basic_features(_silent_signal(), 16000)
        assert features["rms"] == pytest.approx(0.0, abs=1e-6)

    def test_active_signal_has_positive_rms(self):
        features = extract_basic_features(_sine_wave(), 16000)
        assert features["rms"] > 0.0

    def test_silence_ratio_between_0_and_1(self):
        for sig in [_sine_wave(), _silent_signal()]:
            features = extract_basic_features(sig, 16000)
            assert 0.0 <= features["long_silences_ratio"] <= 1.0
