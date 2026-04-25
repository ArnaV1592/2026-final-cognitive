import requests
from pathlib import Path
import subprocess
import argparse

ROOT = Path("/workspace/VoiceMind-V1")
DATA_DIR = ROOT / "data/hindi_dataset"
REPO = "https://github.com/rkritesh210/DementiaBankHindi"

def download():
    repo_dir = DATA_DIR / "repo"
    if repo_dir.exists():
        print("Already downloaded")
        return repo_dir

    subprocess.run(["git", "clone", REPO, str(repo_dir)])
    return repo_dir

def explore(repo_dir):
    print("\nFiles:\n")
    for f in repo_dir.rglob("*"):
        print(f)

def test(api_url):
    files = list((DATA_DIR / "repo").rglob("*.wav"))[:10]

    print(f"\nTesting {len(files)} files...\n")

    for f in files:
        print("Testing:", f.name)
        with open(f, "rb") as audio:
            r = requests.post(api_url + "/screen", files={"file": audio})
            try:
                print(r.json())
            except:
                print("Error:", r.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--api", type=str, default="http://localhost:8000")

    args = parser.parse_args()

    repo_dir = None

    if args.download:
        repo_dir = download()

    if args.explore:
        if not repo_dir:
            repo_dir = DATA_DIR / "repo"
        explore(repo_dir)

    if args.test:
        test(args.api)