# VoiceMind V1.5 — RunPod Deployment Guide

## Prerequisites

- RunPod account (https://runpod.io)
- Git LFS installed locally (`git lfs install`)
- (Optional) Smallest.ai API key for Hindi ASR

---

## Step 1: Create a Pod

1. Go to **Pods → Deploy**
2. Select a GPU template:
   - **Recommended:** A100 40GB or RTX 4090 (fastest inference)
   - **Minimum:** RTX 3090 / A5000 (slower but works)
3. Use the **RunPod PyTorch** official image (CUDA 12.1+, Python 3.10+)
4. Set **Container Disk:** ≥ 40 GB (models cache here)
5. Under **Expose HTTP Ports**, add `8000`

---

## Step 2: Open JupyterLab / SSH Terminal

In your pod, open a terminal.

---

## Step 3: Install ffmpeg

```bash
apt-get update && apt-get install -y ffmpeg
ffmpeg -version   # verify
```

---

## Step 4: Clone the Repo

```bash
cd /workspace
git lfs install
git clone https://github.com/your-org/VoiceMind-V1.git
cd VoiceMind-V1
git lfs pull       # download best_model.pt (~7.6 MB)
```

---

## Step 5: Install Python Requirements

```bash
pip install -r requirements.txt
```

> First run downloads WavLM-large (~1.2 GB) and XLM-RoBERTa-base (~1.1 GB) from HuggingFace.  
> These are cached at `~/.cache/huggingface/`. Subsequent runs are instant.

---

## Step 6: Configure Environment

```bash
cp .env.example .env
nano .env   # add your PULSE_API_KEY if you have one
```

Or export directly:

```bash
export PULSE_API_KEY="sk_your_key_here"
export REFERRAL_THRESH=0.50
```

---

## Step 7: Start the Server

```bash
python app/serve.py
```

You should see:
```
INFO     Loading VoiceMind models …
  Whisper medium …
  WavLM-large …
  XLM-R-base …
  Classifier ✓  temperature=1.5000
INFO     ✅ Model loaded
INFO     Starting VoiceMind V1.5
INFO     Frontend → http://0.0.0.0:8000/
INFO     Docs     → http://0.0.0.0:8000/docs
```

---

## Step 8: Access via RunPod HTTP URL

In RunPod → your pod → **Connect** → **HTTP Service (8000)**.  
Copy the URL (e.g. `https://xxxx-8000.proxy.runpod.net`).

- **Clinical UI:** `https://xxxx-8000.proxy.runpod.net/`
- **API Docs:**   `https://xxxx-8000.proxy.runpod.net/docs`
- **Health:**     `https://xxxx-8000.proxy.runpod.net/health`

---

## Step 9: Keep Server Running (Optional)

Use `screen` or `tmux` to keep the server alive when you close the terminal:

```bash
screen -S voicemind
python app/serve.py
# Ctrl+A, D to detach
# screen -r voicemind to reattach
```

---

## Tip: Persistent Volume

Mount a RunPod **Network Volume** to `/workspace` to persist the HuggingFace model cache and session data between pod restarts. This avoids re-downloading 2+ GB of models on every restart.

---

## Port Overview

| Port | Service |
|---|---|
| 8000 | VoiceMind FastAPI (HTTP) |
| 8888 (default) | JupyterLab |
| 22 | SSH |

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ffmpeg not found` | `apt-get install -y ffmpeg` |
| `CUDA out of memory` | Use a smaller batch or switch to CPU with `DEVICE=cpu` in `predictor.py` |
| `Audio conversion failed` | File may be corrupt or unsupported format — try converting to WAV first |
| Model loads but predictions all 0.5 | Check calibration temperature — should be between 0.5–3.0 |
| Pulse API 401 | Check PULSE_API_KEY is set correctly |
