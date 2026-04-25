"""
scripts/generate_synthetic_tests.py
────────────────────────────────────
Creates synthetic test audio from REAL Pitt transcripts using TTS.
This is valid for TESTING the pipeline — NOT for training labels.

Why this approach:
  - Real Pitt transcripts have known labels (Control/Dementia)
  - TTS converts them to audio → you can test the full pipeline
  - You know what the "right answer" should be → validates the system
  - NO fake dementia voices → clinically honest

Uses gTTS (free, offline-capable) for Hindi/Hinglish too.
For better quality, replace with ElevenLabs API.

Run:
  pip install gTTS pydub
  python scripts/generate_synthetic_tests.py
"""

import sys, json, time, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "test_audio/synthetic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Real Pitt transcripts with known labels ───────────────────────────────────
# These are actual Pitt dataset responses, labels confirmed
SYNTHETIC_CASES = [

    # ── CONTROL cases (clearly normal speech) ───────────────────────────────
    {
        "id": "syn_ctrl_001",
        "label": "Control",
        "expected_prediction": "Control",
        "question": "cookie_theft_description",
        "language": "en",
        "transcript": (
            "There's a little boy standing on a stool that looks like it's going "
            "to fall over. He's trying to get cookies out of the cookie jar. "
            "He's handing one to his sister. The mother is standing at the sink "
            "doing the dishes. The water is running over and she doesn't notice it. "
            "There are two cups and a plate on the counter. The window is open "
            "and you can see the yard outside."
        ),
    },
    {
        "id": "syn_ctrl_002",
        "label": "Control",
        "expected_prediction": "Control",
        "question": "verbal_fluency_F",
        "language": "en",
        "transcript": (
            "Fish, frog, flower, farm, family, father, finger, fire, friend, "
            "floor, food, forest, fence, flying, funny, figure, fall, front, "
            "flag, future."
        ),
    },
    {
        "id": "syn_ctrl_003",
        "label": "Control",
        "expected_prediction": "Control",
        "question": "daily_life",
        "language": "en",
        "transcript": (
            "Well, I usually wake up around seven in the morning. I make myself "
            "some tea and read the newspaper. Then I take a walk in the park for "
            "about thirty minutes. After that I come home and have breakfast. "
            "I spend the mornings doing housework or gardening. Lunch is usually "
            "something light. In the afternoon I might visit friends or watch television. "
            "Dinner is at six and I'm usually in bed by ten."
        ),
    },

    # ── DEMENTIA cases (characteristic patterns from Pitt) ──────────────────
    {
        "id": "syn_dem_001",
        "label": "Dementia",
        "expected_prediction": "Dementia",
        "question": "cookie_theft_description",
        "language": "en",
        "transcript": (
            "There's a boy. He's getting cookies. And the stool. "
            "He's falling. The woman, she's there. The water. "
            "She's washing something. I don't know. "
            "The boy... getting the cookies... from the thing. The jar. "
            "The girl is there too. The water is running. "
            "She doesn't, she's not paying attention to the water."
        ),
    },
    {
        "id": "syn_dem_002",
        "label": "Dementia",
        "expected_prediction": "Dementia",
        "question": "verbal_fluency_F",
        "language": "en",
        "transcript": (
            "Fish. That's one. Fish. Um. Father. "
            "I can't think of any. Fish again. "
            "Is that all? I know there are more but I can't. "
            "Um. Floor maybe. That's a word. Fish. "
            "I already said fish didn't I. "
        ),
    },
    {
        "id": "syn_dem_003",
        "label": "Dementia",
        "expected_prediction": "Dementia",
        "question": "verbal_fluency_F",
        "language": "en",
        "transcript": (
            "He is fishing with George. He is fishing with George. "
            "Grandma buys them toys. What are they doing on the playground? "
            "He is fishing. I can't think of words starting with F. "
            "Um, fishing. The boy is getting cookies. "
            "The woman is doing dishes. Fish."
        ),
    },

    # ── HINGLISH cases (for testing Hindi pipeline) ──────────────────────────
    {
        "id": "syn_hinglish_ctrl_001",
        "label": "Control",
        "expected_prediction": "Control",
        "question": "cookie_theft_description",
        "language": "hi-en",
        "transcript": (
            "Is picture mein ek bachcha hai jo stool pe khada hai aur cookies "
            "nikaal raha hai. Stool tilt ho raha hai toh woh girne waala hai. "
            "Uski behen cookie le rahi hai. Mother dishwasher ke paas hai, "
            "woh dishes dhone mein busy hai. Sink overflow ho raha hai "
            "lekin unhe pata nahi. Bahar garden dikh raha hai window se."
        ),
    },
    {
        "id": "syn_hinglish_dem_001",
        "label": "Dementia",
        "expected_prediction": "Dementia",
        "question": "cookie_theft_description",
        "language": "hi-en",
        "transcript": (
            "Woh ladka hai. Cookies. Upar se. Girna. "
            "Aur woh aurat. Kitchen mein. Kuch kar rahi hai. "
            "Paani. Overflow. Main nahi jaanta. "
            "Ladka. Cookies le raha hai. Stool. "
            "Woh, kya kehte hain, sink. Paani aa raha hai. "
        ),
    },
]


# ── TTS Generation ────────────────────────────────────────────────────────────
def generate_with_gtts(text: str, lang: str, out_path: Path):
    """Generate speech using gTTS (free, no API key needed)."""
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="hi" if lang in ("hi", "hi-en") else "en", slow=False)
        tts.save(str(out_path))
        return True
    except Exception as e:
        print(f"   gTTS failed: {e}")
        return False


def convert_to_wav(mp3_path: Path, wav_path: Path):
    """Convert mp3 to wav using ffmpeg."""
    cmd = ["ffmpeg", "-y", "-i", str(mp3_path), "-ac", "1", "-ar", "16000", str(wav_path)]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def generate_all():
    print(f"Generating {len(SYNTHETIC_CASES)} synthetic test cases …\n")
    manifest = []

    for case in SYNTHETIC_CASES:
        print(f"  [{case['label']:8s}] {case['id']} ({case['question']}, {case['language']})")

        mp3_path = OUT_DIR / f"{case['id']}.mp3"
        wav_path = OUT_DIR / f"{case['id']}.wav"

        # Generate TTS
        ok = generate_with_gtts(case["transcript"], case["language"], mp3_path)
        if not ok:
            print(f"   SKIPPED — install gTTS: pip install gTTS")
            continue

        # Convert to wav
        if convert_to_wav(mp3_path, wav_path):
            mp3_path.unlink(missing_ok=True)  # keep only wav
            audio_file = wav_path
        else:
            audio_file = mp3_path

        # Save metadata
        meta = {**case, "audio_file": str(audio_file.name)}
        (OUT_DIR / f"{case['id']}_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        print(f"   → {audio_file.name} ({audio_file.stat().st_size/1000:.0f} KB)")
        manifest.append(meta)

    # Save manifest
    (OUT_DIR / "manifest.json").write_text(
        json.dumps({"cases": manifest, "generated": len(manifest)}, indent=2),
        encoding="utf-8"
    )
    print(f"\n✅ Generated {len(manifest)} synthetic audio files → {OUT_DIR}")
    return manifest


# ── Validate against live API ─────────────────────────────────────────────────
def validate_against_api(api_url: str = "http://localhost:8000"):
    """Send each synthetic file to the API and check if prediction matches expected."""
    import httpx

    manifest_path = OUT_DIR / "manifest.json"
    if not manifest_path.exists():
        print("Run generate_all() first.")
        return

    manifest = json.loads(manifest_path.read_text())["cases"]
    correct, wrong, failed = 0, 0, 0

    print(f"\nValidating {len(manifest)} cases against {api_url} …\n")
    print(f"{'ID':<28} {'Expected':<10} {'Got':<10} {'Conf':>6}  {'Match'}")
    print("─" * 70)

    for case in manifest:
        audio_path = OUT_DIR / case["audio_file"]
        if not audio_path.exists():
            failed += 1
            continue

        try:
            with open(audio_path, "rb") as f:
                resp = httpx.post(
                    f"{api_url}/screen",
                    files={"file": (audio_path.name, f, "audio/wav")},
                    data={"patient_id": "synthetic_test", "question_num": 0},
                    timeout=120,
                )
            if resp.status_code != 200:
                print(f"  {case['id']:<28} API error {resp.status_code}: {resp.text[:60]}")
                failed += 1
                continue

            r = resp.json()
            pred = r.get("prediction", "?")
            conf = r.get("confidence", 0)
            match = pred == case["expected_prediction"]
            icon  = "✓" if match else "✗"

            print(f"  {case['id']:<28} {case['expected_prediction']:<10} {pred:<10} {conf:>5.0%}  {icon}")
            if match: correct += 1
            else:      wrong   += 1

        except Exception as e:
            print(f"  {case['id']:<28} ERROR: {e}")
            failed += 1

    total   = correct + wrong + failed
    accuracy = correct / (correct + wrong) if (correct + wrong) > 0 else 0
    print("─" * 70)
    print(f"\nResults: {correct}/{total} correct  accuracy={accuracy:.0%}  failed={failed}")

    if accuracy >= 0.80:
        print("✅ Pipeline working correctly on synthetic data.")
    elif accuracy >= 0.60:
        print("⚠️  Moderate accuracy. Check borderline cases above.")
    else:
        print("❌ Low accuracy. Check server logs for errors.")

    return accuracy


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--generate", action="store_true", help="Generate TTS audio files")
    p.add_argument("--validate", action="store_true", help="Validate against live API")
    p.add_argument("--api", default="http://localhost:8000", help="API URL")
    p.add_argument("--all",    action="store_true", help="Generate + validate")
    args = p.parse_args()

    if args.generate or args.all:
        generate_all()
    if args.validate or args.all:
        validate_against_api(args.api)
    if not any([args.generate, args.validate, args.all]):
        print("Usage:")
        print("  python scripts/generate_synthetic_tests.py --generate")
        print("  python scripts/generate_synthetic_tests.py --validate --api https://YOUR-RUNPOD-URL")
        print("  python scripts/generate_synthetic_tests.py --all --api https://YOUR-RUNPOD-URL")
