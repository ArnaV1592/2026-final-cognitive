"""
core/predictor.py — VoiceMind V1.5 (fixed)
──────────────────────────────────────────
Single shared predictor. Models load ONCE. Call .predict() many times.

V1.5 fixes vs the previous version:
  1) predict() now accepts `client_lang` from the frontend.
       en      → Whisper (matches our English training data)
       hi      → Pulse first, Whisper as fallback if Pulse fails
       hi-en   → Pulse first, Whisper as fallback (handles Hinglish)
       auto    → Whisper first, re-route to Pulse if Hindi/Indic detected
  2) predict() returns "_segments" (Whisper word timestamps).
     The clinical scorer needs these to compute real temporal features
     (response latency, pauses, WPM, disfluencies). Without them every
     temporal feature defaulted to 0/placeholder — that's why domain
     scores looked dead even on good audio.
"""
import pickle, time, logging, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import librosa

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000
MAX_DUR   = 90       # seconds — cap audio length
WAVLM_MAX = 30       # seconds — cap clip fed to WavLM (GPU memory)
MIN_DUR   = 5        # seconds — skip audio shorter than this
LABEL_MAP = {0: "Control", 1: "Dementia"}


class VoiceMindPredictor:
    """
    Load once, predict many times.
    Pass `client_lang` to predict() to force Hindi/Hinglish to Pulse.
    """

    # ── Init ─────────────────────────────────────────────────────────────────
    def __init__(self, pulse_api_key: str = "",
                 referral_thresh: float = 0.50,
                 model_dir: Path | None = None):
        import whisper
        from transformers import (AutoFeatureExtractor, WavLMModel,
                                  XLMRobertaModel, XLMRobertaTokenizer)
        from core.model.fusion      import GatedFusion
        from core.model.classifier  import CognitiveClassifier
        from core.model.calibration import TemperatureScaler

        self.pulse_key       = pulse_api_key or ""
        self._pulse_avail    = bool(self.pulse_key)
        self.referral_thresh = float(referral_thresh)
        self.model_dir       = (Path(model_dir) if model_dir else
                                Path(__file__).resolve().parent.parent / "artifacts/model")

        print("  Whisper medium …")
        self._whisper = whisper.load_model("medium", device=DEVICE)

        print("  WavLM-large …")
        self._fe    = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
        self._wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(DEVICE)
        self._wavlm.eval()
        for p in self._wavlm.parameters():
            p.requires_grad = False

        print("  XLM-R-base …")
        self._xt = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        self._xm = XLMRobertaModel.from_pretrained("xlm-roberta-base").to(DEVICE)
        self._xm.eval()
        for p in self._xm.parameters():
            p.requires_grad = False

        fusion    = GatedFusion(1024, 768, 28, 512)
        self._clf = CognitiveClassifier(fusion, 2, 0.2).to(DEVICE)
        self._clf.load_state_dict(
            torch.load(self.model_dir / "best_model.pt",
                       map_location=DEVICE, weights_only=True)
        )
        self._clf.eval()

        with open(self.model_dir / "scaler.pkl", "rb") as f:
            self._scaler = pickle.load(f)

        self._temp = TemperatureScaler()
        self._temp.load(self.model_dir / "calibration_params.json")
        self._temp.to(DEVICE)
        print(f"  Classifier ✓  temperature={self._temp.temperature.item():.4f}")

        if self._pulse_avail:
            log.info("ASR mode: Whisper (English) + Pulse (Hindi/Hinglish)")
        else:
            log.info("ASR mode: Whisper only — set PULSE_API_KEY for Hindi support")

    # ── ASR helpers ──────────────────────────────────────────────────────────
    def _whisper_transcribe(self, audio_path: Path) -> dict:
        """Always-on Whisper transcription with word timestamps."""
        res  = self._whisper.transcribe(
            str(audio_path), language=None, task="transcribe",
            word_timestamps=True, condition_on_previous_text=False,
        )
        text = (res.get("text") or "").strip()
        segs = [
            {"start": s.get("start", 0),
             "end":   s.get("end", 0),
             "text":  (s.get("text") or "").strip(),
             "words": s.get("words", [])}
            for s in res.get("segments", [])
        ]
        return {
            "text":     text,
            "language": res.get("language", "en"),
            "segments": segs,
            "provider": "whisper",
            "cost_usd": 0.0,
        }

    def _pulse_transcribe(self, audio_path: Path) -> dict:
        """Smallest.ai Pulse transcription for Hindi / Hinglish."""
        import httpx
        with open(audio_path, "rb") as f:
            ab = f.read()
        ext  = audio_path.suffix.lower().lstrip(".")
        mime = {"wav": "audio/wav", "mp3": "audio/mpeg",
                "m4a": "audio/mp4", "mp4": "audio/mp4",
                "ogg": "audio/ogg"}.get(ext, "audio/wav")

        resp = httpx.post(
            "https://waves-api.smallest.ai/api/v1/pulse/get_text",
            headers={"Authorization": f"Bearer {self.pulse_key}"},
            files={"file": (audio_path.name, ab, mime)},
            data={"language": "hi-en", "word_timestamps": "true"},
            timeout=120.0,
        )
        resp.raise_for_status()
        pr = resp.json()

        segs = [
            {"start": s.get("start", 0),
             "end":   s.get("end", 0),
             "text":  (s.get("text") or "").strip(),
             "words": s.get("words", [])}
            for s in pr.get("segments", [])
        ]
        dur = segs[-1]["end"] if segs else 0.0
        return {
            "text":     (pr.get("text") or "").strip(),
            "language": pr.get("language", "hi-en"),
            "segments": segs,
            "provider": "pulse",
            "cost_usd": round((dur / 60.0) * 0.006, 5),
        }

    def _transcribe(self, audio_path: Path,
                    client_lang: str = "auto") -> dict:
        """
        Language-aware ASR routing.
          en              → Whisper
          hi or hi-en     → Pulse (Whisper fallback if Pulse errors)
          auto / unknown  → Whisper, then re-route to Pulse if Indic detected
        """
        hint = (client_lang or "auto").lower().strip()

        # 1) Clinician explicitly selected Hindi/Hinglish — go to Pulse first.
        if hint in ("hi", "hi-en") and self._pulse_avail:
            try:
                log.info("Routing to Pulse (clinician selected '%s')", hint)
                return self._pulse_transcribe(audio_path)
            except Exception as e:
                log.warning("Pulse failed (%s) — falling back to Whisper", e)
                return self._whisper_transcribe(audio_path)

        # 2) English or auto — start with Whisper.
        wr = self._whisper_transcribe(audio_path)

        # 3) In auto mode, re-route to Pulse if Whisper output looks Indic.
        if hint == "auto" and self._pulse_avail:
            text = wr["text"]
            lang = wr["language"]
            is_indic = (lang in ("hi", "ur", "pa") or
                        any("\u0900" <= c <= "\u097f" for c in text))
            if is_indic:
                try:
                    log.info("Whisper detected Indic — re-routing to Pulse")
                    return self._pulse_transcribe(audio_path)
                except Exception as e:
                    log.warning("Pulse failed (%s) — using Whisper output", e)

        return wr

    # ── Embeddings ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def _embed_wavlm(self, audio: np.ndarray) -> np.ndarray:
        clip = audio[:WAVLM_MAX * TARGET_SR]
        iv   = self._fe(clip, sampling_rate=TARGET_SR,
                        return_tensors="pt").input_values.to(DEVICE)
        h    = self._wavlm(iv).last_hidden_state
        return h.mean(1).squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def _embed_xlmr(self, text: str) -> np.ndarray:
        enc = self._xt(text or ".", return_tensors="pt",
                       truncation=True, max_length=512).to(DEVICE)
        out = self._xm(**enc)
        return out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy().astype(np.float32)

    # ── Main predict ─────────────────────────────────────────────────────────
    def predict(self, audio_path: Path,
                client_lang: str = "auto") -> dict:
        """
        Run inference on one audio file.

        client_lang:
          "en"     — Whisper only
          "hi"     — Pulse (Whisper fallback)
          "hi-en"  — Pulse (Whisper fallback)
          "auto"   — Whisper, route to Pulse if Indic detected
        """
        from core.features.acoustic   import extract_acoustic
        from core.features.linguistic import extract_linguistic

        t0 = time.time()
        audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        audio = audio[:MAX_DUR * TARGET_SR]
        peak  = float(np.abs(audio).max())
        if peak > 1e-6:
            audio /= peak
        dur = len(audio) / TARGET_SR

        if dur < MIN_DUR:
            return {
                "error":                f"Audio too short ({dur:.1f}s, min {MIN_DUR}s)",
                "prediction":           "SKIP",
                "confidence":           0,
                "referral_recommended": False,
                "transcript":           "",
                "_segments":            [],
            }

        # 1) Transcribe (language-aware)
        tr = self._transcribe(audio_path, client_lang=client_lang)

        # 2) Backbone embeddings
        wav_emb = self._embed_wavlm(audio)
        xlm_emb = self._embed_xlmr(tr["text"])

        # 3) Derived features (acoustic + linguistic, scaled)
        ac  = extract_acoustic(audio, sr=TARGET_SR)
        li  = extract_linguistic(tr["text"], tr["segments"])
        der = np.concatenate([ac.to_ordered_array(),
                              li.to_ordered_array()]).reshape(1, -1)
        der_s = self._scaler.transform(der)

        # 4) Classify + temperature-calibrate
        wav_t = torch.tensor(wav_emb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        xlm_t = torch.tensor(xlm_emb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        der_t = torch.tensor(der_s,   dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            probs = F.softmax(
                self._temp(self._clf(wav_t, xlm_t, der_t)), dim=-1
            ).cpu().numpy()[0]

        pred = int(np.argmax(probs))

        return {
            "prediction":           LABEL_MAP[pred],
            "confidence":           round(float(probs[pred]), 4),
            "P_Control":            round(float(probs[0]),    4),
            "P_Dementia":           round(float(probs[1]),    4),
            "referral_recommended": bool(probs[1] >= self.referral_thresh),
            "transcript":           tr["text"],
            "language_detected":    tr["language"],
            "asr_provider":         tr["provider"],
            "asr_cost_usd":         tr["cost_usd"],
            "audio_duration_s":     round(dur, 1),
            "latency_seconds":      round(time.time() - t0, 2),
            "model_version":        "1.5.0",
            "error":                None,
            "_segments":            tr["segments"],   # ← needed by clinical_scorer
            "disclaimer":           "AI screening only. NOT a medical diagnosis.",
        }
