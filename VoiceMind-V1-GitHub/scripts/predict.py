"""
Predict on a real audio file.
Usage:
  python scripts/predict.py --audio /workspace/test_audio/my_voice.wav
  python scripts/predict.py --audio /workspace/test_audio/my_voice.mp3 --show-transcript
"""
import argparse, sys, json, pickle, time, warnings
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
import librosa, whisper
from transformers import AutoFeatureExtractor, WavLMModel, XLMRobertaModel, XLMRobertaTokenizer

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.features.acoustic   import extract_acoustic
from core.features.linguistic import extract_linguistic
from core.model.fusion        import GatedFusion
from core.model.classifier    import CognitiveClassifier
from core.model.calibration   import TemperatureScaler

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR  = 16000
MODEL_DIR  = ROOT / "artifacts/model"
LABEL_MAP  = {0: "Control", 1: "Dementia"}

print(f"Loading models on {DEVICE} …")
asr    = whisper.load_model("medium", device=DEVICE)
fe     = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
wavlm  = WavLMModel.from_pretrained("microsoft/wavlm-large").to(DEVICE)
wavlm.eval()
for p in wavlm.parameters(): p.requires_grad = False
xt     = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
xm     = XLMRobertaModel.from_pretrained("xlm-roberta-base").to(DEVICE)
xm.eval()
for p in xm.parameters(): p.requires_grad = False

fusion = GatedFusion(1024, 768, 28, 512)
model  = CognitiveClassifier(fusion, 2, 0.2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_DIR/"best_model.pt", map_location=DEVICE, weights_only=True))
model.eval()

with open(MODEL_DIR/"scaler.pkl","rb") as f: scaler = pickle.load(f)
temp = TemperatureScaler(); temp.load(MODEL_DIR/"calibration_params.json"); temp.to(DEVICE)
print("✓ Ready")

def predict(audio_path):
    t0    = time.time()
    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
    if sr != TARGET_SR: audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    audio = audio[:90*TARGET_SR]
    peak  = np.abs(audio).max()
    if peak > 1e-6: audio /= peak

    res  = asr.transcribe(str(audio_path), language=None, task="transcribe",
                          word_timestamps=True, condition_on_previous_text=False)
    text = res["text"].strip()
    segs = [{"start":s["start"],"end":s["end"],"text":s["text"].strip(),
              "words":s.get("words",[])} for s in res.get("segments",[])]

    with torch.no_grad():
        iv   = fe(audio[:30*TARGET_SR], sampling_rate=TARGET_SR, return_tensors="pt").input_values.to(DEVICE)
        wemb = wavlm(iv).last_hidden_state.mean(1).squeeze(0).cpu().numpy()
        enc  = xt(text or ".", return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        xemb = xm(**enc).last_hidden_state[:,0,:].squeeze(0).cpu().numpy()

    ac   = extract_acoustic(audio)
    li   = extract_linguistic(text, segs)
    der  = np.concatenate([ac.to_ordered_array(), li.to_ordered_array()]).reshape(1,-1)
    ders = scaler.transform(der)

    wt = torch.tensor(wemb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    xt_ = torch.tensor(xemb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    dt = torch.tensor(ders, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(temp(model(wt, xt_, dt)), dim=-1).cpu().numpy()[0]

    pred = int(np.argmax(probs))
    return {
        "prediction": LABEL_MAP[pred],
        "confidence": round(float(probs[pred]), 4),
        "P_Control":  round(float(probs[0]), 4),
        "P_Dementia": round(float(probs[1]), 4),
        "referral":   bool(probs[1] >= 0.50),
        "transcript": text,
        "duration_s": round(len(audio)/TARGET_SR, 1),
        "latency_s":  round(time.time()-t0, 2),
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--show-transcript", action="store_true")
    args = p.parse_args()

    r = predict(Path(args.audio))
    print(f"\n{"═"*48}")
    print(f"  Prediction  : {r["prediction"].upper()}")
    print(f"  Confidence  : {r["confidence"]:.1%}")
    print(f"  P(Control)  : {r["P_Control"]:.1%}")
    print(f"  P(Dementia) : {r["P_Dementia"]:.1%}")
    print(f"  Referral    : {"⚠️ YES" if r["referral"] else "✅ No"}")
    print(f"  Duration    : {r["duration_s"]}s | Latency: {r["latency_s"]}s")
    if args.show_transcript: print(f"\n  Transcript:\n  {r["transcript"]}")
    print(f"{"═"*48}")
    print("  ⚠️  AI screening only. NOT a medical diagnosis.")
