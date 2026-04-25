import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.predictor import VoiceMindPredictor

# ── EDIT THESE ────────────────────────────────────────────────────────
SINGLE_FILE = ROOT / "test_audio/test.wav"   # ← change this
PULSE_API_KEY = "sk_cabeaba9cc692ad4d56b22c46f0e66b8" # ← paste your Smallest.ai key here, or leave "" for Whisper
# ─────────────────────────────────────────────────────────────────────

def main():
    vm = VoiceMindPredictor(
        pulse_api_key=PULSE_API_KEY,
        prefer_pulse=bool(PULSE_API_KEY),
    )

    if SINGLE_FILE.exists():
        print(f"Testing: {SINGLE_FILE.name}\n")
        r = vm.predict(SINGLE_FILE)

        if r.get("error"):
            print(f"❌ {r['error']}")
        else:
            print("═"*50)
            print(f"  Prediction  : {r['prediction'].upper()}")
            print(f"  Confidence  : {r['confidence']:.1%}")
            print(f"  P(Control)  : {r['P_Control']:.1%}")
            print(f"  P(Dementia) : {r['P_Dementia']:.1%}")
            print(f"  Referral    : {'⚠️  YES — clinical review recommended' if r['referral_recommended'] else '✅  No'}")
            print(f"  Language    : {r['language_detected']}")
            print(f"  ASR         : {r['asr_provider']}")
            print(f"  Duration    : {r['audio_duration_s']}s | Latency: {r['latency_seconds']}s")
            print(f"\n  Transcript  :")
            print(f"  {r['transcript']}")
            print("═"*50)
            print(f"\n⚠️  {r['disclaimer']}")
    else:
        print(f"File not found: {SINGLE_FILE}")
        print("Upload a file to test_audio/ and update the path above.")

if __name__ == "__main__":
    main()
