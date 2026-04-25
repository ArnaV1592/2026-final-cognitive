
import numpy as np
import librosa
from dataclasses import dataclass

@dataclass
class AcousticFeatures:
    speech_rate: float
    pause_ratio: float
    pitch_variance: float
    mfcc_means: np.ndarray  # shape (20,)

    def to_ordered_array(self) -> np.ndarray:
        return np.concatenate([
            [self.speech_rate, self.pause_ratio, self.pitch_variance],
            self.mfcc_means
        ])

    @staticmethod
    def feature_names():
        return (
            ["speech_rate_syl_per_sec", "pause_ratio", "pitch_variance_hz"] +
            [f"mfcc_mean_{i}" for i in range(20)]
        )

def extract_acoustic(audio: np.ndarray, sr: int = 16000) -> AcousticFeatures:
    duration = len(audio) / sr

    onset_env   = librosa.onset.onset_strength(y=audio, sr=sr)
    onsets      = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    speech_rate = len(onsets) / max(duration, 0.1)

    frame_len   = int(0.025 * sr)
    hop         = int(0.010 * sr)
    energy      = np.array([
        np.sum(audio[i:i+frame_len]**2)
        for i in range(0, len(audio)-frame_len, hop)
    ])
    threshold   = np.percentile(energy, 25)
    pause_ratio = float(np.sum(energy < threshold) / max(len(energy), 1))

    f0, _, _  = librosa.pyin(audio, fmin=50, fmax=500, sr=sr)
    f0v       = f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
    pitch_var = float(np.var(f0v)) if len(f0v) > 1 else 0.0

    mfcc      = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

    return AcousticFeatures(
        speech_rate=speech_rate,
        pause_ratio=pause_ratio,
        pitch_variance=pitch_var,
        mfcc_means=np.mean(mfcc, axis=1),
    )
