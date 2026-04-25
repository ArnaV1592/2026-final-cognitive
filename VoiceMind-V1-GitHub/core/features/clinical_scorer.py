"""
core/features/clinical_scorer.py
─────────────────────────────────
Domain-wise clinical scoring from transcript + Whisper word timestamps.

Returns None if transcript is too short to score (< 8 words).
serve.py then falls back to P_Control-based estimation for that domain.
"""
import re
from dataclasses import dataclass, field
from collections import Counter
import numpy as np

DISFLUENCIES = {
    "um","uh","uhh","umm","er","hmm","hm","ah","like",
    "अ","अं","हं","हम्म","aa","aaa",
}
STOP = {
    "the","a","an","is","are","was","i","my","and","or","it","he",
    "she","they","we","you","of","in","on","at","to","for","that","this",
    "है","हैं","था","के","की","का","में","पर","और","एक","यह","वह","मैं","हम","से",
}
COOKIE_EN = {
    "boy","girl","cookie","cookies","jar","stool","fall","falling","mother",
    "woman","lady","sink","water","overflow","dish","dishes","wash","window",
    "kitchen","plate","cup","standing","stealing","reaching","counter",
}
COOKIE_HI = {
    "लड़का","लड़की","कुकी","बिस्कुट","जार","स्टूल","गिरना","माँ","सिंक",
    "पानी","बर्तन","रसोई","खिड़की","cookie","boy","girl","sink","water",
}

# Minimum words before domain scoring is meaningful
MIN_WORDS_FOR_SCORING = 8


@dataclass
class TemporalFeatures:
    response_latency_s:    float = 0.0
    mean_pause_duration_s: float = 0.0
    max_pause_duration_s:  float = 0.0
    pause_frequency:       float = 0.0
    disfluency_rate:       float = 0.0
    speech_rate_wpm:       float = 80.0
    speech_rate_first_half: float = 80.0
    speech_rate_second_half: float = 80.0
    rate_decay:            float = 1.0


@dataclass
class DomainScores:
    memory:      float = 0.0
    fluency:     float = 0.0
    attention:   float = 0.0
    language:    float = 0.0
    orientation: float = 5.0
    temporal: TemporalFeatures = field(default_factory=TemporalFeatures)
    scored_from_transcript: bool = True  # False = transcript was too short

    @property
    def total(self):
        return self.memory + self.fluency + self.attention + self.language + self.orientation

    @property
    def risk_level(self):
        t = self.total
        if t >= 26: return "Normal"
        if t >= 19: return "Mild impairment"
        if t >= 11: return "Moderate impairment"
        return "Severe impairment"

    def to_dict(self):
        return {
            "total":    round(self.total, 1),
            "max":      30,
            "risk_level": self.risk_level,
            "referral": self.total < 26 or self.temporal.response_latency_s > 4.0,
            "scored_from_transcript": self.scored_from_transcript,
            "domains": {
                "memory":      {"score": round(self.memory, 1),      "max": 8},
                "fluency":     {"score": round(self.fluency, 1),      "max": 7},
                "attention":   {"score": round(self.attention, 1),    "max": 5},
                "language":    {"score": round(self.language, 1),     "max": 5},
                "orientation": {"score": round(self.orientation, 1),  "max": 5},
            },
            "temporal": {
                "response_latency_s":  round(self.temporal.response_latency_s, 2),
                "mean_pause_s":        round(self.temporal.mean_pause_duration_s, 2),
                "max_pause_s":         round(self.temporal.max_pause_duration_s, 2),
                "pauses_per_minute":   round(self.temporal.pause_frequency, 1),
                "disfluency_per_100w": round(self.temporal.disfluency_rate, 1),
                "speech_rate_wpm":     round(self.temporal.speech_rate_wpm, 1),
                "rate_decay":          round(self.temporal.rate_decay, 3),
            },
            "interpretation": _interpret(self),
        }


def _extract_temporal(segments, dur):
    words = []
    for seg in segments:
        for w in seg.get("words", []):
            word = w.get("word", "").strip().lower()
            if word:
                words.append({
                    "word":  word,
                    "start": w.get("start", 0),
                    "end":   w.get("end", 0),
                })
    if not words:
        return TemporalFeatures()

    latency = words[0]["start"]
    pauses  = [
        words[i]["start"] - words[i-1]["end"]
        for i in range(1, len(words))
        if words[i]["start"] - words[i-1]["end"] > 0.25
    ]
    disf       = sum(1 for w in words if w["word"] in DISFLUENCIES)
    speech_dur = max(dur - latency, 1.0)
    wpm        = (len(words) / speech_dur) * 60

    mid    = (words[0]["start"] + words[-1]["end"]) / 2
    fh     = [w for w in words if w["start"] < mid]
    sh     = [w for w in words if w["start"] >= mid]
    wpm_fh = (len(fh) / max(mid - words[0]["start"], 0.1)) * 60
    wpm_sh = (len(sh) / max(words[-1]["end"] - mid, 0.1)) * 60

    return TemporalFeatures(
        response_latency_s     = latency,
        mean_pause_duration_s  = float(np.mean(pauses)) if pauses else 0.0,
        max_pause_duration_s   = float(max(pauses))  if pauses else 0.0,
        pause_frequency        = len(pauses) / max(speech_dur / 60, 0.1),
        disfluency_rate        = (disf / max(len(words), 1)) * 100,
        speech_rate_wpm        = wpm,
        speech_rate_first_half = wpm_fh,
        speech_rate_second_half= wpm_sh,
        rate_decay             = min(wpm_sh / max(wpm_fh, 1.0), 2.0),
    )


def compute_domain_scores(question_num, transcript, segments, audio_duration,
                          recall_words=None):
    """
    Returns DomainScores, or None if transcript is too short to score.
    Caller (serve.py) uses None to fall back to P_Control-based estimation.
    """
    temporal = _extract_temporal(segments, audio_duration)
    tx       = (transcript or "").lower().strip()
    words    = tx.split()
    n        = len(words)

    # If transcript is too short, return None → serve.py uses ML fallback
    if n < MIN_WORDS_FOR_SCORING:
        ds           = DomainScores(temporal=temporal)
        ds.scored_from_transcript = False
        # Still compute temporal features (these come from audio, not text)
        return ds   # scores are all 0 — serve.py will override with ML fallback

    ds = DomainScores(temporal=temporal, scored_from_transcript=True)

    if question_num == 0:        # Picture description → Language
        kw   = sum(1 for k in COOKIE_EN | COOKIE_HI if k in tx)
        ttr  = len(set(words)) / n
        rep  = Counter(words).most_common(1)[0][1] / n if words else 0
        ds.language = min(
            (kw / 5.0) * 2.0
            + (1.5 if ttr > 0.6 else 0.8)
            + (1.0 if temporal.response_latency_s < 3 else 0.3)
            - (0.5 if rep > 0.15 else 0),
            5.0
        )

    elif question_num == 1:      # Verbal fluency → Fluency
        cw  = [w for w in words if w not in STOP and len(w) > 2]
        qty = 3.0 if len(cw) >= 15 else 2.0 if len(cw) >= 10 else 1.0 if len(cw) >= 6 else 0.0
        ttr = len(set(cw)) / max(len(cw), 1)
        div = 2.0 if ttr > 0.85 else 1.5 if ttr > 0.65 else 1.0
        rate= 2.0 if temporal.speech_rate_wpm >= 80 else 1.5 if temporal.speech_rate_wpm >= 55 else 0.5
        ds.fluency = min(qty + div + rate, 7.0)

    elif question_num == 2:      # Serial subtraction → Attention
        nums  = [int(x) for x in re.findall(r"\d+", tx) if int(x) <= 100]
        steps = sum(
            1 for i in range(len(nums)-1)
            if abs(nums[i] - nums[i+1] - 7) <= 1
        ) if len(nums) >= 2 else 0
        ds.attention = min(
            steps * 1.0 + (2.0 if temporal.response_latency_s < 3 else 0.5),
            5.0
        )

    elif question_num == 4:      # Delayed recall → Memory (most important)
        rw      = recall_words or ["apple", "table", "penny"]
        recalled= sum(1 for w in rw if w.lower() in tx)
        lat_s   = 2.0 if temporal.response_latency_s < 2 else                   1.5 if temporal.response_latency_s < 4 else 0.5
        ds.memory = min(
            recalled * 1.67 + lat_s
            + (1.0 if temporal.disfluency_rate < 5 else 0.0),
            8.0
        )

    return ds


def _interpret(ds):
    items = []
    if ds.memory < 4:
        items.append({"domain":"Memory","icon":"✗",
                      "status":"Impaired — poor delayed recall. Strongest Alzheimer's indicator."})
    elif ds.memory < 6:
        items.append({"domain":"Memory","icon":"⚠","status":"Mildly reduced recall."})
    else:
        items.append({"domain":"Memory","icon":"✓","status":"Normal recall."})

    if ds.fluency < 4:
        items.append({"domain":"Verbal fluency","icon":"✗",
                      "status":"Low word generation — frontal lobe concern."})
    elif ds.fluency < 5:
        items.append({"domain":"Verbal fluency","icon":"⚠","status":"Mildly reduced."})
    else:
        items.append({"domain":"Verbal fluency","icon":"✓","status":"Normal."})

    if ds.attention < 3:
        items.append({"domain":"Attention","icon":"✗",
                      "status":"Impaired — possible vascular or delirium component."})
    else:
        items.append({"domain":"Attention","icon":"✓","status":"Adequate."})

    if ds.language < 3:
        items.append({"domain":"Language","icon":"⚠",
                      "status":"Reduced coherence in picture description."})
    else:
        items.append({"domain":"Language","icon":"✓","status":"Normal coherence."})

    t = ds.temporal
    if t.response_latency_s > 5:
        items.append({"domain":"Response latency","icon":"⚠",
                      "status": f"Elevated ({t.response_latency_s:.1f}s) — word-finding difficulty."})
    if t.disfluency_rate > 10:
        items.append({"domain":"Disfluency","icon":"⚠",
                      "status": f"High ({t.disfluency_rate:.0f}/100 words)."})
    if not ds.scored_from_transcript:
        items.append({"domain":"Note","icon":"ℹ",
                      "status":"Transcript too short for text scoring — domain scores estimated from voice model only."})
    return items
