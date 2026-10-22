"""Pipeline orchestration: feature extraction -> training -> live inference.

Three presets:

  ``--preset accuracy``  (the default) the configuration that maximises the
                         accuracy on REAL-TIME sentences.  It is the one the
                         project is tuned for: the student-teacher signal
                         (so the CTC head never collapses on the blank), the
                         three-source mix (so the 228 classes are all known),
                         the per-layer learning rates (so the pretrained layers
                         survive), the strided front-end (2.3x faster, so
                         there are more epochs in the same time) and the
                         prefix beam search with the language model at
                         inference time (so the decoder does not insert the
                         signs the corpus never contains).

  ``--preset fast``      the same pipeline with the greedy decoder and no
                         language model, for a machine that cannot afford the
                         beam search.

  ``--preset minimal``   only the steps that are needed to refresh the
                         artefacts (the split manifest and the language model).

Every step can also be run on its own with ``--only <step>``.
"""
import argparse
import os
import subprocess
import sys

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BACKEND_DIR, "models")

# The shared vocabulary + split manifest (every phase reads them).
COMMON = ["--min_samples_per_class", "14", "--max_samples_per_class", "15"]


def accuracy_steps():
    """The maximum-accuracy configuration."""
    return [
        ("split", ["scripts/save_phase4_split.py", "--canonical"]),
        ("lm", ["scripts/build_word_lm.py"]),

        # --- the word model (the class knowledge of all 228 classes) -------
        ("pretrain", ["training/pretrainer.py"]),
        ("phase2", ["training/phase2_trainer.py"]),
        ("phase3", ["training/phase3_trainer.py"]),
        ("phase4", ["training/phase4_trainer.py"]),
        ("phase5", ["training/phase5_trainer.py"]),
        ("phase6", ["training/phase6_trainer.py"]),

        # --- the sentence model --------------------------------------------
        # NO student-teacher distillation here on purpose: the only candidate
        # teacher is the Phase-6 word classifier (81.9% top-1, word-only, and
        # not trained properly), and distilling a sentence model from a weak
        # word classifier would make the sentence model inherit its mistakes.
        # The student-teacher loss stays available as an OPT-IN experiment
        # (--kd_weight), it is simply not part of the recommended recipe.
        #
        # The accuracy comes from the three-source mix (the class knowledge of
        # all 228 classes + the alignment + the real domain), the hardened
        # anti-collapse CTC loss, the per-layer learning rates and the prefix
        # beam search with the language model at inference time.
        ("ctc", ["training/train_conformer_ctc.py",
                 "--stage", "1",
                 "--stride", "2", "--gru_expand", "after",
                 "--layer_lrs",
                 "front=3e-4,expander=3e-4,gru=1e-5,transformer=1e-5,"
                 "tcnn=3e-5,conformer=1.5e-4,head=1.5e-4",
                 "--freeze_transformer_epochs", "3",
                 "--beam_width", "25", "--beam_alpha", "0.5", "--beam_lm_mode", "bigram",
                 "--escape_patience", "6"]),
        ("sentence", ["training/train_synthetic_sentences.py",
                      "--beam_width", "25", "--beam_alpha", "0.5"]),
        ("real", ["training/train_real_sentences.py",
                  "--mix", "1", "--mix_weights", "0.40", "0.35", "0.25",
                  # 8 out-of-scope words: 1296 real clips / 921 sentences / 156
                  # classes (the default 5 throws away 27% of the real corpus;
                  # 12 is the ceiling at 1411 clips / 161 classes).
                  "--max_out_scope", "8", "--word_cap", "20",
                  "--beam_width", "25", "--beam_alpha", "0.5"]),
    ]


def fast_steps():
    return [
        ("split", ["scripts/save_phase4_split.py", "--canonical"]),
        ("lm", ["scripts/build_word_lm.py"]),
        ("pretrain", ["training/pretrainer.py"]),
        ("phase2", ["training/phase2_trainer.py"]),
        ("phase3", ["training/phase3_trainer.py"]),
        ("phase4", ["training/phase4_trainer.py"]),
        ("phase5", ["training/phase5_trainer.py"]),
        ("phase6", ["training/phase6_trainer.py"]),
        ("ctc", ["training/train_conformer_ctc.py", "--stage", "1",
                 "--stride", "2", "--gru_expand", "after"]),
        ("sentence", ["training/train_synthetic_sentences.py"]),
        ("real", ["training/train_real_sentences.py", "--mix", "1",
                  "--max_out_scope", "5", "--word_cap", "15"]),
    ]


def minimal_steps():
    return [
        ("split", ["scripts/save_phase4_split.py", "--canonical"]),
        ("lm", ["scripts/build_word_lm.py"]),
    ]


PRESETS = {"accuracy": accuracy_steps, "fast": fast_steps, "minimal": minimal_steps}


def run_script(script_path, extra_args=()):
    print("\n" + "=" * 70)
    print(f"Running {script_path} {' '.join(extra_args)}".strip())
    print("=" * 70)
    cmd = [sys.executable, script_path, *extra_args]
    result = subprocess.run(cmd, cwd=BACKEND_DIR)
    if result.returncode != 0:
        print(f"Error running {script_path} (exit code {result.returncode})")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="MookVani pipeline orchestrator")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="accuracy",
                        help="The configuration to run (default: accuracy)")
    parser.add_argument("--only", help="Run only this step")
    parser.add_argument("--list", action="store_true", help="List the steps and exit")
    args = parser.parse_args()

    steps = PRESETS[args.preset]()

    if args.list:
        for name, cmd in steps:
            print(f"{name:<10} {' '.join(cmd)}")
        return

    selected = [(n, c) for n, c in steps if not args.only or n == args.only]
    for name, cmd in selected:
        path = os.path.join(BACKEND_DIR, cmd[0])
        if not os.path.exists(path):
            print(f"[SKIP] {name}: {cmd[0]} does not exist")
            continue
        run_script(cmd[0], cmd[1:])

    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()
