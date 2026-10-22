"""Pipeline orchestration: feature extraction -> training.

Every step is a real, existing script of this repository.  Run a single step
with ``--only <step>``, or run the whole chain with no argument.
"""
import argparse
import os
import subprocess
import sys

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = {
    # name: (script, extra args)
    "extract_words": ("scripts/extraction.py", []),
    "extract_sentences": ("scripts/extract_real_sentences.py", []),
    "pretrain": ("training/pretrainer.py", []),
    "word": ("training/train_reconformer_word.py", []),
    "sentence": ("training/train_synthetic_sentences.py", []),
}


def run_script(script_path, extra_args=()):
    print("\n" + "=" * 60)
    print(f"Running {script_path} {' '.join(extra_args)}".strip())
    print("=" * 60)
    cmd = [sys.executable, script_path, *extra_args]
    result = subprocess.run(cmd, cwd=BACKEND_DIR)
    if result.returncode != 0:
        print(f"Error running {script_path} (exit code {result.returncode})")
        sys.exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MookVani pipeline orchestrator")
    parser.add_argument("--only", choices=list(STEPS.keys()),
                        help="Run only this step (default: the whole chain)")
    args = parser.parse_args()

    selected = [args.only] if args.only else list(STEPS.keys())
    for name in selected:
        script, extra = STEPS[name]
        path = os.path.join(BACKEND_DIR, script)
        if not os.path.exists(path):
            print(f"[SKIP] {name}: {script} does not exist")
            continue
        run_script(script, extra)

    print("\nAll tasks completed successfully!")
