# VoiceMind V1.5 — Voice Biomarker Model for Cognitive Decline

> **AI-powered, multilingual (English + Hindi) dementia screening from voice recordings.**  
> Research prototype — **NOT** a clinical diagnostic tool.

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Model Performance](#model-performance)
4. [Quickstart](#quickstart)
5. [API Reference](#api-reference)
6. [Project Structure](#project-structure)
7. [Configuration & Environment Variables](#configuration--environment-variables)
8. [Running on RunPod / Cloud GPU](#running-on-runpod--cloud-gpu)
9. [Scripts: Batch & Single Inference](#scripts-batch--single-inference)
10. [Training Notes](#training-notes)
11. [Limitations & Disclaimer](#limitations--disclaimer)

---

## Overview

VoiceMind V1.5 is a **multimodal AI model** that detects cognitive decline (Alzheimer's / dementia vs. healthy control) from short voice recordings (≥ 2 seconds, optimal 30–90 seconds).

### What it does
- Accepts audio in any browser format (WAV, MP3, WebM, OGG, M4A)
- Transcribes via **Whisper medium** (English) or **Smallest.ai Pulse** (Hindi / Hinglish)
- Extracts **acoustic** (MFCCs, pitch, pause ratio, speech rate) and **linguistic** (type-token ratio, lexical entropy, semantic coherence) features
- Fuses **WavLM-large** audio embeddings + **XLM-RoBERTa-base** text embeddings + 28 derived features through a Gated Fusion layer
- Returns a calibrated probability: `P(Control)` / `P(Dementia)` plus MMSE-style **domain scores** (Memory, Fluency, Attention, Language, Orientation)
- Saves every session to disk for clinician labelling and future fine-tuning

---

## Architecture

```
Audio File (WAV/MP3/WebM)
        │
        ▼
   FFmpeg → 16 kHz mono WAV
        │
   ┌────┴─────────────────────────────────────┐
   │                                           │
   ▼                                           ▼
Whisper medium                        WavLM-large
(ASR + language detect)           (acoustic embedding)
   │  Hindi/Hinglish?                          │
   │  ─── yes → Pulse API                      │
   │                                           │
   ▼                                           ▼
  Transcript                           wav_emb [1024]
        │
        ▼
 XLM-RoBERTa-base
(text embedding)
        │
        ▼
  xlm_emb [768]

  ┌──────────────────────────────────────┐
  │  Acoustic features  [23]             │
  │  Linguistic features [5]             │
  │  → derived_vec [28] (scaled)         │
  └──────────────────────────────────────┘
             │
             ▼
     GatedFusion(1024+768+28 → 512)
             │
             ▼
  CognitiveClassifier(512 → 256 → 2)
             │
  TemperatureScaler (post-hoc calibration)
             │
             ▼
  P(Control)   P(Dementia)
             +
  Domain Scores (MMSE-style, clinical_scorer.py)
```

### Key Components

| File | Purpose |
|---|---|
| `core/predictor.py` | Main inference class — loads all models once, call `.predict()` many times |
| `core/model/fusion.py` | `GatedFusion` — element-wise gated projection of all three embedding streams |
| `core/model/classifier.py` | `CognitiveClassifier` — two-layer MLP head + `LabelSmoothingCE` for training |
| `core/model/calibration.py` | `TemperatureScaler` — post-hoc probability calibration (LBFGS fit) |
| `core/features/acoustic.py` | `extract_acoustic()` — speech rate, pause ratio, pitch variance, 20 MFCCs |
| `core/features/linguistic.py` | `extract_linguistic()` — TTR, lexical entropy, pronoun consistency, syntax depth |
| `core/features/clinical_scorer.py` | `compute_domain_scores()` — MMSE-style domain scores from transcript + word timestamps |
| `app/serve.py` | FastAPI server — `/screen`, `/label`, `/sessions`, `/health` endpoints |
| `app/frontend/index.html` | Self-contained single-page clinical UI |

---

## Model Performance

Training dataset: **DementiaBank Pitt Corpus** (English), subset of DementiaBank Hindi  
Task: Binary classification — `Control` vs `Dementia`

| Metric | Value |
|---|---|
| Cross-validation accuracy | ~82–85% (5-fold) |
| Calibration temperature | 1.5 (clamped, safety-bounded to [0.1, 5.0]) |
| Referral threshold | P(Dementia) ≥ 0.50 |
| Inference latency (CPU) | ~15–45 s / file |
| Inference latency (GPU A100) | ~3–8 s / file |

> See `artifacts/model/cv_results.json` for full per-fold metrics.

---

## Quickstart

### Prerequisites
- Python 3.10+
- `ffmpeg` installed system-wide (`apt-get install -y ffmpeg` / `brew install ffmpeg`)
- CUDA GPU recommended (CPU works but is slow for WavLM-large)
- Git LFS installed (`git lfs install`) — needed to pull `best_model.pt`

### 1. Clone & pull model weights

```bash
git clone https://github.com/your-org/VoiceMind-V1.git
cd VoiceMind-V1
git lfs pull          # downloads artifacts/model/best_model.pt (~7.6 MB)
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — add your PULSE_API_KEY if you have one (optional, for Hindi ASR)
```

### 4. Start the server

```bash
python app/serve.py
```

Open `http://localhost:8000` in your browser for the clinical UI, or `http://localhost:8000/docs` for the Swagger API explorer.

---

## API Reference

### `POST /screen`
Run inference on a voice recording.

**Form fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | UploadFile | required | Audio file (WAV, MP3, WebM, OGG, M4A) |
| `patient_id` | str | `""` | Patient identifier — triggers session save |
| `question_num` | int | `0` | MMSE question number (0=picture, 1=fluency, 2=attention, 4=recall) |
| `clinician` | str | `""` | Clinician name (for session metadata) |
| `session_id` | str | `""` | Group question answers by session |
| `recall_words` | str | `"[]"` | JSON list, e.g. `'["Apple","Table","Penny"]'` — for Q4 recall scoring |
| `language` | str | `"en"` | Hint: `"en"` / `"hi"` / `"hi-en"` |

**Response:**
```json
{
  "prediction": "Control",
  "confidence": 0.82,
  "P_Control": 0.82,
  "P_Dementia": 0.18,
  "referral_recommended": false,
  "transcript": "The boy is stealing cookies from the jar...",
  "language_detected": "en",
  "asr_provider": "whisper",
  "asr_cost_usd": 0.0,
  "audio_duration_s": 42.3,
  "latency_seconds": 6.1,
  "model_version": "1.5.0",
  "domain_scores": {
    "total": 24.5,
    "max": 30,
    "risk_level": "Mild impairment",
    "domains": {
      "memory":      {"score": 6.0, "max": 8},
      "fluency":     {"score": 5.0, "max": 7},
      "attention":   {"score": 4.0, "max": 5},
      "language":    {"score": 4.5, "max": 5},
      "orientation": {"score": 5.0, "max": 5}
    }
  },
  "disclaimer": "AI screening only. NOT a medical diagnosis."
}
```

### `GET /health`
Returns model loaded status, version, and session count.

### `POST /label`
Clinician adds MMSE score and ground truth after assessment. Used to build labelled training data.

### `GET /sessions`
Lists all saved patient sessions grouped by `patient_id`.

---

## Project Structure

```
VoiceMind-V1/
│
├── app/
│   ├── serve.py              # FastAPI server
│   └── frontend/
│       └── index.html        # Clinical web UI (single file)
│
├── core/
│   ├── predictor.py          # VoiceMindPredictor — main inference class
│   ├── asr/
│   │   ├── router.py         # ASR language routing logic
│   │   └── pulse_client.py   # Smallest.ai Pulse API wrapper
│   ├── features/
│   │   ├── acoustic.py       # Acoustic feature extraction
│   │   ├── linguistic.py     # Linguistic feature extraction
│   │   └── clinical_scorer.py  # MMSE-style domain scoring
│   ├── model/
│   │   ├── fusion.py         # GatedFusion layer
│   │   ├── classifier.py     # CognitiveClassifier + LabelSmoothingCE
│   │   └── calibration.py    # TemperatureScaler
│   └── pipeline/
│       └── __init__.py
│
├── scripts/
│   ├── predict.py            # Single-file CLI inference
│   ├── batch_inference.py    # Batch over a folder of audio files
│   ├── single_inference.py   # Minimal single-file example
│   ├── generate_synthetic_tests.py  # TTS-based synthetic test generation
│   └── use_hindi_dataset.py  # Hindi dataset processing
│
├── artifacts/
│   └── model/
│       ├── best_model.pt         # Trained weights (Git LFS)
│       ├── scaler.pkl            # StandardScaler for derived features
│       ├── calibration_params.json  # Temperature scaling params
│       └── cv_results.json       # Cross-validation metrics
│
├── config/
│   └── feature_schema.json   # Feature names, dims, label map
│
├── data/
│   └── pilot_sessions/       # Session audio + JSON (gitignored, local only)
│
├── test_audio/               # Put your test .wav/.mp3 files here (gitignored)
│
├── docs/
│   ├── ARCHITECTURE.md       # Detailed technical architecture
│   ├── TRAINING.md           # Training procedure & dataset notes
│   └── RUNPOD_SETUP.md       # RunPod deployment guide
│
├── .env.example              # Environment variable template
├── .gitattributes            # Git LFS config
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Configuration & Environment Variables

Copy `.env.example` → `.env` and fill in:

| Variable | Required | Description |
|---|---|---|
| `PULSE_API_KEY` | No | Smallest.ai API key for Hindi/Hinglish ASR. Without it, Whisper is used for all languages. |
| `REFERRAL_THRESH` | No | Float 0–1.0, default `0.50`. P(Dementia) threshold for referral flag. |

---

## Running on RunPod / Cloud GPU

See [`docs/RUNPOD_SETUP.md`](docs/RUNPOD_SETUP.md) for full instructions.

**Quick path:**

```bash
# 1. Spin up a RunPod pod with PyTorch image (CUDA 12.x)
# 2. Clone repo + git lfs pull
# 3. pip install -r requirements.txt
# 4. apt-get install -y ffmpeg
# 5. Set environment variables (PULSE_API_KEY if needed)
# 6. python app/serve.py
# 7. Expose port 8000 as HTTP service in RunPod settings
```

---

## Scripts: Batch & Single Inference

### Batch over a folder
```bash
# Set API key (optional)
export PULSE_API_KEY="sk_..."

# Run over all audio in test_audio/
python scripts/batch_inference.py
# Saves results.csv and results.json in test_audio/
```

### Single file CLI
```bash
python scripts/predict.py --audio test_audio/my_recording.wav --show-transcript
```

---

## Training Notes

See [`docs/TRAINING.md`](docs/TRAINING.md) for details. Key points:

- **Dataset:** DementiaBank Pitt Corpus (cookie-theft picture description task)
- **Backbone:** WavLM-large + XLM-RoBERTa-base — **frozen** during training (feature extractor only)
- **Trained parameters:** GatedFusion + CognitiveClassifier head only (~600K params)
- **Loss:** `LabelSmoothingCE` (smoothing=0.1) with class-weight balancing
- **Calibration:** Post-hoc temperature scaling (validation set, LBFGS)
- **Training platform:** RunPod A100 40GB, ~3–4 hours

---

## Limitations & Disclaimer

> ⚠️ **This is a research prototype, not a certified medical device.**
> - Results must NOT be used for clinical diagnosis without physician review
> - Trained on limited datasets; performance may degrade on unseen demographics
> - Hindi/Hinglish performance is lower than English (smaller training corpus)
> - Short recordings (< 10 seconds) reduce accuracy significantly

---

*VoiceMind V1.5 — Built on RunPod · Model: WavLM-large + XLM-RoBERTa-base · ASR: Whisper medium + Smallest.ai Pulse*
