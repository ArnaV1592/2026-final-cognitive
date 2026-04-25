# VoiceMind V1.5 — Training Notes

## Overview

The VoiceMind classifier head was trained on the **DementiaBank Pitt Corpus** (cookie-theft picture description task), binary label: `Control (0)` vs `Dementia (1)`.

The backbone models (WavLM-large, XLM-RoBERTa-base) are **frozen** — only the GatedFusion layer and CognitiveClassifier head are trained.

---

## Dataset

| Source | Type | Labels |
|---|---|---|
| DementiaBank Pitt Corpus | English audio + transcripts | Control / Dementia |
| DementiaBank Hindi (partial) | Hindi audio | Control / Dementia |

> The full DementiaBank corpus requires IRB agreement: https://dementia.talkbank.org/

### Preprocessing

- Audio resampled to 16 kHz mono (librosa)
- Normalised to peak amplitude
- Clipped to 90 seconds max (30s for WavLM embedding)
- Short samples < 2 seconds skipped

### Pre-extracted embeddings (not in repo — training artefacts)

During training on RunPod, embeddings were pre-extracted and cached as numpy arrays:
- `artifacts/wav_embs.npy`   — WavLM-large CLS embeddings [N, 1024]
- `artifacts/xlm_embs.npy`   — XLM-R CLS embeddings [N, 768]
- `artifacts/derived.npy`    — Scaled acoustic+linguistic features [N, 28]
- `artifacts/labels.npy`     — Ground-truth labels [N]
- `artifacts/speaker_ids.npy` — Speaker IDs for leave-one-speaker-out CV

These files are **not committed to Git** (too large) — they live only on the RunPod training volume.

---

## Training Procedure

### Loss Function
```python
LabelSmoothingCE(smoothing=0.1, weight=class_weights)
# class_weights computed from label distribution to handle imbalance
```

### Optimizer
- Adam, lr=1e-3, weight_decay=1e-4
- ReduceLROnPlateau scheduler (patience=5)

### Cross-Validation
- 5-fold stratified CV, grouped by speaker (speaker-independent split)
- Best model saved by validation accuracy per fold

### Calibration
After training:
```python
T = TemperatureScaler()
T.fit(val_logits, val_labels)
T.save("artifacts/model/calibration_params.json")
```

---

## Metrics (from cv_results.json)

See `artifacts/model/cv_results.json` for full per-fold accuracy, F1, and AUC.

---

## Fine-tuning with New Data

Clinicians can label new sessions via `POST /label` which saves ground truth to `data/pilot_sessions/`.

To fine-tune:
1. Re-extract embeddings for new sessions
2. Concatenate with original training data
3. Re-run training script (not included in this repo — contact team)
4. Replace `artifacts/model/best_model.pt` and `scaler.pkl`

---

## Hardware Used

- **Platform:** RunPod
- **GPU:** NVIDIA A100 40GB
- **Training time:** ~3–4 hours (pre-extracted embeddings, training head only)
- **Environment:** CUDA 12.1, Python 3.10, PyTorch 2.2
