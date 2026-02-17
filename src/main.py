import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

def run_step(rel_path: str) -> None:
    path = ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Missing pipeline step file: {path}")
    cmd = [PY, str(path)]
    print("\n== Running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"Step failed ({rc}): {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ingestion", action="store_true")
    args = ap.parse_args()

    if not args.skip_ingestion:
        run_step("src/ingestion/fetch_chicago_food_inspections_by_date.py")

    run_step("src/cleaning/make_silver_food_inspections_from_jsonl_parts.py")
    run_step("src/features/make_gold_features.py")
    run_step("src/scoring/score_and_export.py")

    print("\nPipeline complete.")

if __name__ == "__main__":
    main()
