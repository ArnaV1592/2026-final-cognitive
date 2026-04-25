
import logging, time
from pathlib import Path
from dataclasses import dataclass

log = logging.getLogger(__name__)

@dataclass
class TranscriptResult:
    text: str
    language: str          # "en" | "hi" | "hi-en"
    language_family: str   # "english" | "hindi" | "mixed"
    segments: list
    provider: str          # "whisper" | "pulse"
    cost_usd: float
    duration_s: float

# Words that strongly signal Hindi/mixed speech
_HINDI_SIGNALS = {
    "aap", "main", "mera", "tera", "haan", "nahi", "theek", "accha",
    "toh", "matlab", "lekin", "aur", "kya", "kaise", "kyun", "yeh",
    "woh", "unka", "mujhe", "humko", "apna", "bahut", "thoda", "abhi",
    "pehle", "baad", "agar", "sab", "kuch", "sirf", "bilkul"
}

def _detect_language_from_text(text: str) -> tuple[str, str]:
    """
    Fast heuristic language detection from transcript text.
    Returns (language_code, language_family).
    Used AFTER transcription to decide if we should retry with Pulse.
    """
    words = text.lower().split()
    if not words:
        return "en", "english"

    # Count Hindi-signal words
    hindi_hits = sum(1 for w in words if w in _HINDI_SIGNALS)
    hindi_ratio = hindi_hits / len(words)

    # Check for Devanagari script
    has_devanagari = any(
        "\u0900" <= c <= "\u097f" for c in text
    )

    if has_devanagari or hindi_ratio > 0.15:
        if hindi_ratio < 0.5 and not has_devanagari:
            return "hi-en", "mixed"
        return "hi", "hindi"

    return "en", "english"


class ASRRouter:
    """
    Smart ASR router. Call transcribe() on any audio file.
    Automatically picks the best engine based on language.

    Rules:
      English audio  →  Whisper  (model was trained on Whisper transcripts)
      Hindi audio    →  Pulse    (Whisper is poor at Hindi)
      Mixed audio    →  Pulse    (handles Hinglish natively)
      Pulse missing  →  Whisper  (fallback for everything)
    """

    def __init__(self, whisper_model, pulse_api_key: str = ""):
        self._whisper     = whisper_model
        self._pulse_key   = pulse_api_key
        self._pulse_avail = bool(pulse_api_key)

        if self._pulse_avail:
            log.info("ASR router: Whisper (English) + Pulse (Hindi/Mixed) enabled")
        else:
            log.info("ASR router: Whisper only (set PULSE_API_KEY for Hindi support)")

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        """
        Step 1: Run Whisper to get a fast transcript + language hint.
        Step 2: Check if audio is Hindi/mixed.
        Step 3: If Hindi and Pulse available → re-transcribe with Pulse.
        Step 4: Return the best result.
        """
        # Always run Whisper first (fast, good language ID)
        whisper_result = self._whisper_transcribe(audio_path)
        lang_code, lang_family = _detect_language_from_text(whisper_result["text"])

        # If English → use Whisper result directly (best for our trained model)
        if lang_family == "english" or not self._pulse_avail:
            return TranscriptResult(
                text=whisper_result["text"],
                language=lang_code,
                language_family=lang_family,
                segments=whisper_result["segments"],
                provider="whisper",
                cost_usd=0.0,
                duration_s=whisper_result["duration"],
            )

        # Hindi or Mixed → try Pulse for better accuracy
        log.info(f"  Language detected: {lang_family} — routing to Pulse ASR")
        try:
            pulse_result = self._pulse_transcribe(audio_path)
            return TranscriptResult(
                text=pulse_result["text"],
                language=pulse_result["language"],
                language_family=lang_family,
                segments=pulse_result["segments"],
                provider="pulse",
                cost_usd=pulse_result["cost_usd"],
                duration_s=pulse_result["duration"],
            )
        except Exception as e:
            log.warning(f"  Pulse failed ({e}) — falling back to Whisper")
            return TranscriptResult(
                text=whisper_result["text"],
                language=lang_code,
                language_family=lang_family,
                segments=whisper_result["segments"],
                provider="whisper_fallback",
                cost_usd=0.0,
                duration_s=whisper_result["duration"],
            )

    def _whisper_transcribe(self, audio_path: Path) -> dict:
        res  = self._whisper.transcribe(
            str(audio_path), language=None, task="transcribe",
            word_timestamps=True, condition_on_previous_text=False,
        )
        segs = [{"start": s["start"], "end": s["end"],
                 "text": s["text"].strip(), "words": s.get("words", [])}
                for s in res.get("segments", [])]
        dur  = segs[-1]["end"] if segs else 0.0
        return {"text": res["text"].strip(), "language": res.get("language","en"),
                "segments": segs, "duration": dur}

    def _pulse_transcribe(self, audio_path: Path) -> dict:
        import httpx
        COST_PER_MIN = 0.006
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        ext  = audio_path.suffix.lower().lstrip(".")
        mime = {"wav":"audio/wav","mp3":"audio/mpeg",
                "m4a":"audio/mp4","ogg":"audio/ogg"}.get(ext,"audio/wav")
        resp = httpx.post(
            "https://waves-api.smallest.ai/api/v1/pulse/get_text",
            headers={"Authorization": f"Bearer {self._pulse_key}"},
            files={"file": (audio_path.name, audio_bytes, mime)},
            data={"language": "hi-en", "word_timestamps": "true"},
            timeout=120.0,
        )
        resp.raise_for_status()
        r    = resp.json()
        segs = [{"start": s.get("start",0), "end": s.get("end",0),
                 "text": s.get("text","").strip(), "words": s.get("words",[])}
                for s in r.get("segments", [])]
        dur  = segs[-1]["end"] if segs else 0.0
        return {"text": r.get("text","").strip(), "language": r.get("language","hi-en"),
                "segments": segs, "duration": dur,
                "cost_usd": round((dur/60.0)*COST_PER_MIN, 5)}