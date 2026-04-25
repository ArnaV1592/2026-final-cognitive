# VoiceMind Architecture — Technical Deep Dive

## 1. High-Level Pipeline

```
                    ┌──────────────────────┐
                    │  Audio Input          │
                    │  (WAV/MP3/WebM/OGG)  │
                    └──────────┬───────────┘
                               │
                         FFmpeg convert
                         16 kHz, mono
                               │
               ┌───────────────┼──────────────────┐
               │               │                  │
               ▼               ▼                  ▼
        ┌──────────┐   ┌──────────────┐   ┌──────────────────┐
        │  Whisper │   │  WavLM-large │   │ Acoustic Feature │
        │  medium  │   │  (frozen)    │   │ Extractor        │
        │  (ASR)   │   │              │   │ (librosa)        │
        └────┬─────┘   └──────┬───────┘   └────────┬─────────┘
             │                │                    │
    Hindi?   │         wav_emb [1024]     [23 features]
    → Pulse  │                │            speech_rate
             │           pool over         pause_ratio
             ▼            time             pitch_var
        Transcript               MFCCs [20]
             │
             ▼
     XLM-RoBERTa-base      ┌──────────────────────┐
     (frozen)              │ Linguistic Features   │
             │             │ TTR, entropy,         │
     xlm_emb [768]         │ coherence,            │
                           │ pronoun_inconsistency │
                           │ syntax_depth          │
                           └──────────┬────────────┘
                                      │
                               [5 features]
                                      │
                               concat w/ acoustic
                               → derived [28]
                               StandardScaler.transform()

     ┌──────────────────────────────────────────────────────┐
     │                  GatedFusion                         │
     │   x = cat(wav_emb, xlm_emb, derived)  [1820]        │
     │   out = LayerNorm( Linear(x) * Sigmoid(Linear(x)) )  │
     │   → fused [512]                                      │
     └────────────────────────┬─────────────────────────────┘
                              │
                     CognitiveClassifier
                     Linear(512→256) → ReLU → Dropout(0.2)
                     Linear(256→2)
                              │
                    TemperatureScaler
                    logits / T  (T clamped [0.1, 5.0])
                              │
                       Softmax → [P_Control, P_Dementia]
                              │
                    clinical_scorer.py
                    → Domain scores (MMSE-style)
```

---

## 2. Feature Dimensions

| Stream | Extractor | Dim | Notes |
|---|---|---|---|
| Acoustic embedding | WavLM-large | 1024 | Mean-pool over time, max 30s clip |
| Text embedding | XLM-RoBERTa-base | 768 | CLS token, max 512 tokens |
| Acoustic derived | librosa | 23 | speech_rate + pause_ratio + pitch_var + 20 MFCCs |
| Linguistic derived | custom | 5 | TTR + entropy + coherence + pronoun_inc + syntax_depth |
| **Total derived** | — | **28** | Scaled by `StandardScaler` fitted on training set |

---

## 3. Acoustic Features (23)

| Feature | Description |
|---|---|
| `speech_rate_syl_per_sec` | Onset events / duration (proxy for syllable rate) |
| `pause_ratio` | Fraction of 10ms frames below 25th-percentile energy |
| `pitch_variance_hz` | Variance of f0 (pyin estimator, 50–500 Hz) |
| `mfcc_mean_0..19` | Mean of each of 20 MFCCs across full recording |

---

## 4. Linguistic Features (5)

| Feature | Description |
|---|---|
| `semantic_coherence` | Cosine similarity between consecutive Whisper segments (0 if no sentence embedder) |
| `lexical_entropy` | Shannon entropy of word frequency distribution |
| `type_token_ratio` | Unique words / total words |
| `pronoun_inconsistency` | Unique pronouns / total pronouns (high = inconsistent pronoun use) |
| `syntax_tree_depth` | Proxy: mean words-per-segment / 5, capped at 5 |

---

## 5. GatedFusion

```python
class GatedFusion(nn.Module):
    def __init__(self, wavlm_dim=1024, xlmr_dim=768, derived_dim=28, fused_dim=512):
        total     = wavlm_dim + xlmr_dim + derived_dim   # 1820
        self.proj = nn.Linear(total, fused_dim)
        self.gate = nn.Sequential(nn.Linear(total, fused_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(fused_dim)

    def forward(self, wav_emb, xlm_emb, derived):
        x = torch.cat([wav_emb, xlm_emb, derived], dim=-1)
        return self.norm(self.proj(x) * self.gate(x))
```

The element-wise gating allows the model to learn which input dimensions are most relevant for each output neuron — effectively a soft attention over the concatenated modalities.

---

## 6. Clinical Domain Scoring

`clinical_scorer.py` maps Whisper word timestamps + transcript text to MMSE-style scores for 4 question types:

| Q# | Task | Domain scored | Max |
|---|---|---|---|
| 0 | Cookie-theft picture description | Language | 5 |
| 1 | Verbal fluency (category words) | Fluency | 7 |
| 2 | Serial subtraction (100-7) | Attention | 5 |
| 4 | Delayed word recall | Memory | 8 |

Orientation (max 5) defaults to 5.0 unless further context is provided.

**Fallback:** If transcript < 8 words (too short to score), `clinical_scorer` sets `scored_from_transcript=False` and `serve.py` estimates domain scores from `P_Control` (ML-based fallback).

---

## 7. Temperature Calibration

Post-hoc temperature scaling is applied after training:

```python
T = TemperatureScaler()
T.fit(val_logits, val_labels)   # LBFGS optimisation
# T.temperature is clamped to [0.1, 5.0] for safety
# T > 4.0 at fit time → clamped to 2.0 (protects from over-softening)
```

Calibrated probabilities are more reliable for the referral threshold decision.

---

## 8. ASR Routing

```
                Whisper transcribes (language=None)
                         │
               language in {"hi","ur","pa"}
               OR any Devanagari char in text?
                    │              │
                   YES             NO
                    ▼              ▼
          PULSE_API_KEY set?   Use Whisper output
                    │
               YES     NO
                ▼       ▼
            Pulse   Use Whisper
            API     output
           (hi-en)
```

Pulse fallback: if Pulse API call fails for any reason, Whisper output is used.

---

## 9. Session Persistence

Every `/screen` call with a non-empty `patient_id` saves:
- Original audio file → `data/pilot_sessions/<session_id>_q<N>.<ext>`
- Full JSON result → `data/pilot_sessions/<session_id>_q<N>_result.json`

Clinicians then call `POST /label` to add `mmse_score` and `ground_truth` — creating a labelled dataset for future fine-tuning.
