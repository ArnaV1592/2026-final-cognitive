import sys, os
import json, csv, time
import pandas as pd
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── EDIT THESE ────────────────────────────────────────────────────────
# Set PULSE_API_KEY via environment variable:
#   export PULSE_API_KEY="sk_..."   (Linux/Mac)
#   $env:PULSE_API_KEY="sk_..."     (Windows PowerShell)
# Leave unset to use Whisper-only mode (English).
PULSE_API_KEY  = os.environ.get("PULSE_API_KEY", "")
TEST_AUDIO_DIR = ROOT / "test_audio"    # folder with your audio files
SHOW_TRANSCRIPTS = True    # set False to hide transcripts in output
# ─────────────────────────────────────────────────────────────────────

from core.predictor import VoiceMindPredictor

def main():
    # Load models ONCE
    vm = VoiceMindPredictor(
        pulse_api_key=PULSE_API_KEY,
        prefer_pulse=bool(PULSE_API_KEY),
    )
    print(f"\nASR mode: {'Smallest.ai Pulse (Hindi/Hinglish)' if PULSE_API_KEY else 'Whisper (English)'}")
    print(f"Test folder: {TEST_AUDIO_DIR}")

    # Collect all audio files
    audio_files = sorted([
        f for f in TEST_AUDIO_DIR.rglob("*")
        if f.suffix.lower() in (".wav", ".mp3", ".m4a", ".ogg", ".flac")
    ])

    if not audio_files:
        print(f"\n⚠️  No audio files found in {TEST_AUDIO_DIR}")
        print("   Upload .wav or .mp3 files there.")
        return

    print(f"\nFound {len(audio_files)} audio files. Processing…\n")
    print("═" * 60)

    results  = []
    total_cost = 0.0

    for i, fp in enumerate(audio_files, 1):
        print(f"[{i:2d}/{len(audio_files)}] {fp.name}")
        r = vm.predict(fp)
        results.append(r)

        if r.get("error"):
            print(f"         SKIPPED: {r['error']}")
            continue

        icon = "🔴 DEMENTIA" if r["prediction"] == "Dementia" else "🟢 CONTROL "
        flag = "⚠️  REFER" if r["referral_recommended"] else "   OK"
        print(f"         {icon}  conf={r['confidence']:.1%}  "
              f"P(Dem)={r['P_Dementia']:.1%}  {flag}  "
              f"[{r['asr_provider']}]  {r['latency_seconds']}s")

        if SHOW_TRANSCRIPTS and r["transcript"]:
            preview = r["transcript"][:120].replace("\n"," ")
            print(f"         Transcript: {preview}{'…' if len(r['transcript'])>120 else ''}")

        total_cost += r.get("asr_cost_usd", 0)
        print()

    # ── Summary ───────────────────────────────────────────────────────
    valid   = [r for r in results if not r.get("error") and r["prediction"] != "SKIP"]
    dem     = [r for r in valid if r["prediction"] == "Dementia"]
    ctrl    = [r for r in valid if r["prediction"] == "Control"]
    refs    = [r for r in valid if r["referral_recommended"]]
    skipped = [r for r in results if r.get("error") or r["prediction"] == "SKIP"]

    print("═" * 60)
    print("BATCH SUMMARY")
    print(f"  Files processed : {len(valid)}")
    print(f"  Skipped (short) : {len(skipped)}")
    print(f"  Dementia        : {len(dem)}")
    print(f"  Control         : {len(ctrl)}")
    print(f"  Referrals       : {len(refs)}")
    if total_cost > 0:
        print(f"  ASR cost        : ${total_cost:.4f} USD")
    if valid:
        avg_lat = sum(r["latency_seconds"] for r in valid) / len(valid)
        print(f"  Avg latency     : {avg_lat:.1f}s per file")

    # ── Save to CSV ───────────────────────────────────────────────────
    csv_out = TEST_AUDIO_DIR / "results.csv"
    cols    = ["file","prediction","confidence","P_Control","P_Dementia",
               "referral_recommended","language_detected","asr_provider",
               "audio_duration_s","latency_seconds","transcript"]

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(valid)

    json_out = TEST_AUDIO_DIR / "results.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved:")
    print(f"   {csv_out}   ← open in Excel / share with co-founder")
    print(f"   {json_out}  ← full JSON with transcripts")
    print(f"\n⚠️  All results are AI screening only. NOT medical diagnosis.")

if __name__ == "__main__":
    main()
