"""
core/asr/pulse_client.py
Smallest.ai Pulse ASR — handles Hindi, English, Hinglish.
"""
import httpx, asyncio, time
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ASRResult:
    text: str
    language: str
    segments: list
    duration: float
    provider: str
    cost_usd: float

PULSE_API_URL  = "https://waves-api.smallest.ai/api/v1/pulse/get_text"
COST_PER_MIN   = 0.006

def transcribe_pulse(audio_path: Path, api_key: str) -> ASRResult:
    if not api_key or api_key == "YOUR_PULSE_API_KEY":
        raise ValueError("Pulse API key not set")

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    ext = audio_path.suffix.lower().lstrip(".")
    mime = {"wav":"audio/wav","mp3":"audio/mpeg","mp4":"audio/mp4",
            "m4a":"audio/mp4","ogg":"audio/ogg"}.get(ext, "audio/wav")

    resp = httpx.post(
        PULSE_API_URL,
        headers={"Authorization": f"Bearer " + api_key},
        files={"file": (audio_path.name, audio_bytes, mime)},
        data={
            "model": "pulse-v1",
            "language": "hi-en",
            "word_timestamps": "true",
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    r = resp.json()

    segments = []
    for s in r.get("segments", []):
        segments.append({
            "start": s.get("start", 0),
            "end":   s.get("end", 0),
            "text":  s.get("text", "").strip(),
            "words": s.get("words", []),
            "language": s.get("language", "hi-en"),
        })

    duration = segments[-1]["end"] if segments else 0.0
    return ASRResult(
        text=r.get("text", "").strip(),
        language=r.get("language", "hi-en"),
        segments=segments,
        duration=duration,
        provider="pulse",
        cost_usd=round((duration / 60.0) * COST_PER_MIN, 5),
    )
