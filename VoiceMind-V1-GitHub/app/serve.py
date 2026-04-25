"""
app/serve.py  —  VoiceMind V1.5
Run:  python app/serve.py
Expose port 8000 in RunPod → HTTP Service
"""
import sys, shutil, uuid, os, json, subprocess, logging
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, Response
import json as _json
import uvicorn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP_DIR     = Path("/tmp/voicemind");     TMP_DIR.mkdir(exist_ok=True)
SESSION_DIR = ROOT / "data/pilot_sessions"; SESSION_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("voicemind")

_vm = {}   # global predictor — loaded once at startup


# ── Audio conversion: any browser format → 16kHz mono WAV ───────────────────
def to_wav(src: Path) -> Path:
    if src.suffix.lower() == ".wav":
        return src
    dst = src.with_suffix(".wav")
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
           "-sample_fmt", "s16", str(dst)]
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg: " + r.stderr.decode(errors="replace")[:200])
    return dst


# ── Startup ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    from core.predictor import VoiceMindPredictor
    log.info("Loading VoiceMind models …")
    _vm["p"] = VoiceMindPredictor(
        pulse_api_key=os.environ.get("PULSE_API_KEY", ""),
        referral_thresh=float(os.environ.get("REFERRAL_THRESH", "0.50")),
    )
    log.info("✅ Model loaded")
    yield


app = FastAPI(title="VoiceMind V1.5", version="1.5.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── Suppress favicon spam ────────────────────────────────────────────────────
@app.get("/favicon.ico", include_in_schema=False)
async def _fav():
    return JSONResponse({}, status_code=204)


# ── Serve frontend ───────────────────────────────────────────────────────────
FRONTEND = ROOT / "app/frontend/index.html"

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    if FRONTEND.exists():
        return HTMLResponse(FRONTEND.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>VoiceMind running. Save index.html to app/frontend/</h2>"
                        "<p><a href='/docs'>API docs</a></p>")


# ── Health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "model_loaded":  bool(_vm),
        "model_version": "1.5.0",
        "sessions_saved": len(list(SESSION_DIR.glob("*_result.json"))),
        "asr_modes":     "whisper+pulse" if os.environ.get("PULSE_API_KEY") else "whisper-only",
    }


# ── Main inference endpoint ──────────────────────────────────────────────────
@app.post("/screen")
async def screen(
    file:         UploadFile = File(...),
    patient_id:   str  = Form(default=""),
    question_num: int  = Form(default=0),
    clinician:    str  = Form(default=""),
    session_id:   str  = Form(default=""),
    recall_words: str  = Form(default="[]"),   # JSON list, e.g. '["Apple","Table","Penny"]'
    language:     str  = Form(default="en"),   # "en" | "hi" | "hi-en"
):
    if not _vm:
        raise HTTPException(503, "Model not loaded. Wait 30s and retry.")

    ext      = Path(file.filename or "audio.webm").suffix.lower() or ".webm"
    tmp_orig = TMP_DIR / f"{uuid.uuid4().hex}{ext}"
    tmp_wav  = None

    try:
        # Save upload
        with open(tmp_orig, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Check file is not empty
        if tmp_orig.stat().st_size < 1000:
            raise HTTPException(422, "Audio file is empty or too short.")

        # Convert to WAV (handles webm/ogg/m4a from browsers)
        try:
            tmp_wav = to_wav(tmp_orig)
        except RuntimeError as e:
            raise HTTPException(422, f"Audio conversion failed: {e}. "
                                     "Make sure ffmpeg is installed: apt-get install -y ffmpeg")

        # Run inference (returns dict with prediction + transcript + segments)
        result = _vm["p"].predict(tmp_wav)

        if result.get("error"):
            raise HTTPException(422, result["error"])

        # ── Domain scoring (clinical layer) ──────────────────────────────
        try:
            from core.features.clinical_scorer import compute_domain_scores
            rw = json.loads(recall_words) if recall_words else []
            ds = compute_domain_scores(
                question_num   = question_num,
                transcript     = result.get("transcript", ""),
                segments       = result.get("_segments", []),
                audio_duration = result.get("audio_duration_s", 60.0),
                recall_words   = rw or None,
            )
            domain_dict = ds.to_dict()

            # If transcript was too short, override domain scores with
            # ML model's P_Control-based estimate.  This prevents the
            # clinical scorer from zeroing every domain on empty text.
            if not ds.scored_from_transcript:
                pc = result.get("P_Control", 0.5)
                q  = question_num
                if q == 0:   domain_dict["domains"]["language"]["score"]  = round(pc * 5, 1)
                elif q == 1: domain_dict["domains"]["fluency"]["score"]   = round(pc * 7, 1)
                elif q == 2: domain_dict["domains"]["attention"]["score"] = round(pc * 5, 1)
                elif q == 4: domain_dict["domains"]["memory"]["score"]    = round(pc * 8, 1)
                # Recalculate total
                d = domain_dict["domains"]
                domain_dict["total"] = round(
                    d["memory"]["score"] + d["fluency"]["score"] +
                    d["attention"]["score"] + d["language"]["score"] +
                    d["orientation"]["score"], 1)
                log.info("Q%d: short transcript → ML fallback (P_Control=%.2f)", q, pc)

            result["domain_scores"] = domain_dict
            result["temporal"]      = domain_dict.get("temporal", {})
        except Exception as e:
            log.warning("Domain scoring skipped: %s", e)
            result["domain_scores"] = {}
            result["temporal"]      = {}

        # Remove internal field before returning
        result.pop("_segments", None)

        # Truncate transcript in response to prevent LocalProtocolError.
        # Full transcript is preserved in the session JSON file below.
        _tx = result.get("transcript", "")
        if len(_tx) > 1500:
            result["transcript"] = _tx[:1500] + " … [truncated in response; full version in session file]"
            result["transcript_truncated"] = True
        else:
            result["transcript_truncated"] = False
        log.info("Transcript length: %d chars → response: %d chars",
                 len(_tx), len(result.get("transcript", "")))

        # ── Save session for later labelling ─────────────────────────────
        if patient_id:
            sid  = session_id or f"{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            adst = SESSION_DIR / f"{sid}_q{question_num}{ext}"
            mdst = SESSION_DIR / f"{sid}_q{question_num}_result.json"
            shutil.copy(tmp_orig, adst)
            meta = {
                "session_id":   sid,
                "patient_id":   patient_id,
                "question_num": question_num,
                "language":     language,
                "clinician":    clinician,
                "timestamp":    datetime.now().isoformat(),
                "audio_file":   adst.name,
                "recall_words": json.loads(recall_words) if recall_words else [],
                "prediction":   result,
                "mmse_score":   None,
                "ground_truth": None,
            }
            mdst.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            log.info("Saved: %s Q%d → %s %.0f%%",
                     patient_id, question_num,
                     result.get("prediction"), result.get("confidence", 0)*100)

        return Response(
            content=_json.dumps(result, ensure_ascii=False, default=str),
            media_type="application/json"
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Inference error")
        raise HTTPException(500, f"Inference error: {e}")
    finally:
        tmp_orig.unlink(missing_ok=True)
        if tmp_wav and tmp_wav != tmp_orig:
            tmp_wav.unlink(missing_ok=True)


# ── Label a session after clinical assessment ────────────────────────────────
@app.post("/label")
async def label(
    session_id:   str = Form(...),
    question_num: int = Form(default=0),
    mmse_score:   int = Form(...),
    ground_truth: str = Form(default=""),
    notes:        str = Form(default=""),
):
    """Clinician adds MMSE score after assessment. Creates labelled training data."""
    path = SESSION_DIR / f"{session_id}_q{question_num}_result.json"
    if not path.exists():
        raise HTTPException(404, f"Session {session_id} Q{question_num} not found.")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if not ground_truth:
        ground_truth = "Control" if mmse_score >= 24 else "MCI" if mmse_score >= 18 else "Dementia"
    meta.update({
        "mmse_score":   mmse_score,
        "ground_truth": ground_truth,
        "notes":        notes,
        "labelled_at":  datetime.now().isoformat(),
    })
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    pred = meta.get("prediction", {}).get("prediction", "?")
    log.info("Labelled %s Q%d: pred=%s truth=%s MMSE=%d", session_id, question_num,
             pred, ground_truth, mmse_score)
    return {"ok": True, "prediction": pred, "ground_truth": ground_truth,
            "match": pred == ground_truth}


# ── List saved sessions ──────────────────────────────────────────────────────
@app.get("/sessions")
async def sessions():
    data = {}
    for f in sorted(SESSION_DIR.glob("*_result.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        pid = m.get("patient_id", "?")
        if pid not in data:
            data[pid] = {"patient_id": pid, "language": m.get("language","en"),
                         "questions": [], "labelled": False}
        data[pid]["questions"].append({
            "q":           m.get("question_num"),
            "prediction":  m.get("prediction", {}).get("prediction"),
            "confidence":  m.get("prediction", {}).get("confidence"),
            "mmse":        m.get("mmse_score"),
            "ground_truth":m.get("ground_truth"),
        })
        if m.get("ground_truth"):
            data[pid]["labelled"] = True
    return {"total": len(data), "sessions": list(data.values())}


# ── Start ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting VoiceMind V1.5")
    log.info("Frontend → http://0.0.0.0:8000/")
    log.info("Docs     → http://0.0.0.0:8000/docs")
    uvicorn.run("app.serve:app", host="0.0.0.0", port=8000, reload=False)