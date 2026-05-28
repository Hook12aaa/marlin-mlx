import json
import os
import subprocess
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).parent
VIDEO_DIR = EVAL_DIR / "carebench" / "videos"
METADATA = EVAL_DIR / "carebench" / "json" / "metadata.json"
ENGINE = Path(__file__).parent.parent / "build" / "marlin-caption"
MODEL = Path(__file__).parent.parent / "marlin-2b-ref"

def run_caption(video_path: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    result = subprocess.run(
        [str(ENGINE), str(MODEL), video_path],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, timeout=300
    )
    t1 = time.perf_counter()
    output = result.stdout.strip()
    wall = t1 - t0
    return output, wall


def main():
    with open(METADATA) as f:
        data = json.load(f)

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    if not VIDEO_DIR.exists():
        zip_path = EVAL_DIR / "carebench" / "videos" / "videos.zip"
        if zip_path.exists():
            print(f"Extracting {zip_path}...")
            subprocess.run(["unzip", "-q", str(zip_path), "-d", str(VIDEO_DIR)])

    results = []
    for i, entry in enumerate(data[:n]):
        video_file = VIDEO_DIR / entry["video"]
        if not video_file.exists():
            print(f"[{i+1}/{n}] SKIP {entry['video']} (not found)")
            continue

        print(f"[{i+1}/{n}] {entry['video']}...", end=" ", flush=True)
        output, wall = run_caption(str(video_file))
        print(f"{wall:.1f}s")

        results.append({
            "video": entry["video"],
            "category": entry["category"],
            "ground_truth": entry["caption"][:200],
            "prediction": output[:200],
            "wall_s": wall,
        })

    out_path = EVAL_DIR / "carebench_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{len(results)} videos processed. Results: {out_path}")
    if results:
        mean_wall = sum(r["wall_s"] for r in results) / len(results)
        print(f"Mean wall time: {mean_wall:.1f}s per video")


if __name__ == "__main__":
    main()
