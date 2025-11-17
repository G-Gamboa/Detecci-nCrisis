from typing import Tuple, Dict

import librosa
import numpy as np


def load_audio(path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    signal, sr = librosa.load(path, sr=sample_rate)
    return signal, sr


def extract_basic_features(
    signal: np.ndarray,
    sample_rate: int
) -> Dict[str, float]:
    rms = np.mean(librosa.feature.rms(y=signal))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=signal))
    frame_length = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)
    energy = librosa.feature.rms(
        y=signal,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    silence_threshold = np.percentile(energy, 20)
    long_silences_ratio = float(np.mean(energy < silence_threshold))
    return {
        "rms": float(rms),
        "zcr": float(zcr),
        "long_silences_ratio": long_silences_ratio,
    }
