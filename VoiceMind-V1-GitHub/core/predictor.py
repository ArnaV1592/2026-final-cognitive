"""
core/predictor.py  —  VoiceMind V1.5
Models load ONCE. Call predict() many times.
Language routing: English → Whisper, Hindi/Mixed → Pulse (if key set).
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
MAX_DUR   = 90
WAVLM_MAX = 30
LABEL_MAP = {0: "Control", 1: "Dementia"}
MIN_DUR   = 2    # seconds


class VoiceMindPredictor:

    def __init__(self, model_dir=None, pulse_api_key="", referral_thresh=0.50):
        self.model_dir       = Path(model_dir) if model_dir else \
                               Path(__file__).resolve().parent.parent / "artifacts/model"
        self.pulse_key       = pulse_api_key
        self.referral_thresh = referral_thresh
        self._pulse_avail    = bool(pulse_api_key)

        print(f"[VoiceMind] Loading on {DEVICE} …")
        t0 = time.time()
        self._load_whisper()
        self._load_backbones()
        self._load_classifier()
        print(f"[VoiceMind] ✓ Ready in {time.time()-t0:.1f}s")

    # ── Model loading ────────────────────────────────────────────────────────
    def _load_whisper(self):
        import whisper
        print("  Whisper medium …")
        self._whisper = whisper.load_model("medium", device=DEVICE)

    def _load_backbones(self):
        from transformers import AutoFeatureExtractor, WavLMModel
        from transformers import XLMRobertaModel, XLMRobertaTokenizer
        print("  WavLM-large …")
        self._fe    = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
        self._wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(DEVICE)
        self._wavlm.eval()
        for p in self._wavlm.parameters(): p.requires_grad = False
        print("  XLM-R-base …")
        self._xt = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        self._xm = XLMRobertaModel.from_pretrained("xlm-roberta-base").to(DEVICE)
        self._xm.eval()
        for p in self._xm.parameters(): p.requires_grad = False

    def _load_classifier(self):
        from core.model.fusion      import GatedFusion
        from core.model.classifier  import CognitiveClassifier
        from core.model.calibration import TemperatureScaler
        fusion = GatedFusion(1024, 768, 28, 512)
        self._clf = CognitiveClassifier(fusion, 2, 0.2).to(DEVICE)
        self._clf.load_state_dict(
            torch.load(self.model_dir / "best_model.pt",
                       map_location=DEVICE, weights_only=True))
        self._clf.eval()
        with open(self.model_dir / "scaler.pkl", "rb") as f:
            self._scaler = pickle.load(f)
        from core.model.calibration import TemperatureScaler
        self._temp = TemperatureScaler()
        self._temp.load(self.model_dir / "calibration_params.json")
        self._temp.to(DEVICE)
        print(f"  Classifier ✓  temperature={self._temp.temperature.item():.4f}")

    # ── ASR ──────────────────────────────────────────────────────────────────
    def _transcribe(self, audio_path: Path) -> dict:
        """
        Routes to Pulse if Hindi detected, otherwise uses Whisper.
        Returns: {text, language, segments, provider, cost_usd}
        """
        # Always run Whisper first for language detection
        res   = self._whisper.transcribe(
            str(audio_path), language=None, task="transcribe",
            word_timestamps=True, condition_on_previous_text=False,
        )
        text  = res["text"].strip()
        segs  = [{"start": s["start"], "end": s["end"],
                  "text":  s["text"].strip(),
                  "words": s.get("words", [])}
                 for s in res.get("segments", [])]
        lang  = res.get("language", "en")

        # Detect if Hindi/mixed from Whisper's language tag
        is_indic = lang in ("hi", "ur", "pa") or \
                   any("\u0900" <= c <= "\u097f" for c in text)

        # Route to Pulse if available and speech is Hindi/mixed
        if is_indic and self._pulse_avail:
            try:
                import httpx
                with open(audio_path, "rb") as f:
                    ab = f.read()
                ext  = audio_path.suffix.lower().lstrip(".")
                mime = {"wav":"audio/wav","mp3":"audio/mpeg",
                        "m4a":"audio/mp4","ogg":"audio/ogg"}.get(ext,"audio/wav")
                resp = httpx.post(
                    "https://waves-api.smallest.ai/api/v1/pulse/get_text",
                    headers={"Authorization": f"Bearer {self.pulse_key}"},
                    files={"file": (audio_path.name, ab, mime)},
                    data={"language": "hi-en", "word_timestamps": "true"},
                    timeout=120.0,
                )
                resp.raise_for_status()
                pr   = resp.json()
                p_segs = [{"start": s.get("start",0), "end": s.get("end",0),
                           "text": s.get("text","").strip(),
                           "words": s.get("words",[])}
                          for s in pr.get("segments", [])]
                dur = p_segs[-1]["end"] if p_segs else 0.0
                return {
                    "text":     pr.get("text","").strip(),
                    "language": pr.get("language","hi-en"),
                    "segments": p_segs,
                    "provider": "pulse",
                    "cost_usd": round((dur/60)*0.006, 5),
                }
            except Exception as e:
                log.warning("Pulse failed (%s), using Whisper", e)

        return {
            "text":     text,
            "language": lang,
            "segments": segs,
            "provider": "whisper",
            "cost_usd": 0.0,
        }

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
    def predict(self, audio_path: Path) -> dict:
        from core.features.acoustic   import extract_acoustic
        from core.features.linguistic import extract_linguistic

        t0    = time.time()
        audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        audio = audio[:MAX_DUR * TARGET_SR]
        peak  = np.abs(audio).max()
        if peak > 1e-6: audio /= peak
        dur   = len(audio) / TARGET_SR

        if dur < MIN_DUR:
            return {
                "error":                f"Audio too short ({dur:.1f}s, min {MIN_DUR}s)",
                "prediction":           "SKIP",
                "confidence":           0,
                "referral_recommended": False,
                "transcript":           "",
            }

        # Transcribe
        tr = self._transcribe(audio_path)

        # Backbone embeddings
        wav_emb = self._embed_wavlm(audio)
        xlm_emb = self._embed_xlmr(tr["text"])

        # Derived features
        ac  = extract_acoustic(audio, sr=TARGET_SR)
        li  = extract_linguistic(tr["text"], tr["segments"])
        der = np.concatenate([ac.to_ordered_array(),
                              li.to_ordered_array()]).reshape(1, -1)
        der_s = self._scaler.transform(der)

        # Classify + calibrate
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
            "confidence":           round(float(probs[pred]),  4),
            "P_Control":            round(float(probs[0]),     4),
            "P_Dementia":           round(float(probs[1]),     4),
            "referral_recommended": bool(probs[1] >= self.referral_thresh),
            "transcript":           tr["text"],
            "language_detected":    tr["language"],
            "asr_provider":         tr["provider"],
            "asr_cost_usd":         tr["cost_usd"],
            "audio_duration_s":     round(dur, 1),
            "latency_seconds":      round(time.time() - t0, 2),
            "model_version":        "1.5.0",
            "error":                None,
            "disclaimer":           "AI screening only. NOT a medical diagnosis.",
            # Internal — used by clinical_scorer in serve.py, not sent to client
            "_segments":            tr["segments"],
        }