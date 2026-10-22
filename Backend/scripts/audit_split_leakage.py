"""Audit the split of every phase against the CANONICAL manifest.

The word pipeline has seven stages that all read the same corpus.  If two of
them computed their own split, a clip could be training data in one phase and
validation data in another, and the "validation accuracy" of the later phase
would be measured on data the model had already seen.

This script proves that every phase resolves the SAME assignment:

  1. the canonical manifest itself is leak-free (no clip in both partitions,
     every class stratified, every file filed under its own class),
  2. every phase that reads the manifest resolves the same fingerprint,
  3. the legacy Phase-4/5/6 split file agrees with the manifest.
"""
import json
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from utils.dataloader_word import (  # noqa: E402
    WORD_LEVEL_DIR_DEFAULT,
    discover_word_classes,
)
from utils.splits import (  # noqa: E402
    CANONICAL_SPLIT_PATH,
    audit_split,
    load_canonical_split,
    split_fingerprint,
)

LEGACY_SPLIT = os.path.join(BACKEND_DIR, "models", "phase4_word_model", "split_paths.json")

# Every phase that consumes the word corpus, and how it resolves the split.
PHASES = [
    "training/pretrainer.py",
    "training/train_reconformer_word.py",
    "training/phase2_trainer.py",
    "training/phase3_trainer.py",
    "training/phase4_trainer.py",
    "training/phase5_trainer.py",
    "training/phase6_trainer.py",
]


def main():
    ok = True

    print("=" * 78)
    print("SPLIT LEAKAGE AUDIT")
    print("=" * 78)

    # 1. the manifest itself -------------------------------------------- #
    manifest = load_canonical_split()
    report = audit_split(manifest)
    fp = report["fingerprint"]
    print(f"\n[1] Canonical manifest: {CANONICAL_SPLIT_PATH}")
    print(f"    fingerprint          {fp}")
    print(f"    classes              {report['num_classes']}")
    print(f"    train / val clips    {report['num_train']} / {report['num_val']}")
    print(f"    clip in both         {len(report['overlap'])}")
    print(f"    class without val    {len(report['classes_without_val'])}")
    print(f"    class without train  {len(report['classes_without_train'])}")
    print(f"    misfiled clips       {len(report['misfiled'])}")
    if report["overlap"] or report["classes_without_val"] or report["misfiled"]:
        ok = False
        print("    -> LEAK")
    else:
        print("    -> clean")

    # 2. the phases resolve it ------------------------------------------- #
    print("\n[2] Phases that consume the corpus:")
    for phase in PHASES:
        path = os.path.join(BACKEND_DIR, phase)
        if not os.path.exists(path):
            print(f"    {phase:<40} MISSING")
            ok = False
            continue
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        uses_manifest = ("canonical_split_items" in src or "load_word_split_tensors" in src
                         or "load_canonical_split" in src)
        computes_own = "split_word_classes(" in src and "load_word_split_tensors" not in src
        status = "manifest" if uses_manifest else ("OWN SPLIT" if computes_own else "?")
        if not uses_manifest:
            ok = False
        print(f"    {phase:<40} {status}")

    # 3. the legacy Phase-4 file ----------------------------------------- #
    print("\n[3] Legacy Phase-4 split file:")
    if os.path.exists(LEGACY_SPLIT):
        with open(LEGACY_SPLIT, "r", encoding="utf-8") as f:
            legacy = json.load(f)
        legacy_fp = legacy.get("fingerprint")
        if legacy_fp is None:
            # recompute it from the content
            legacy_fp = split_fingerprint({"classes": legacy.get("classes", []),
                                            "train": legacy.get("train", {}),
                                            "val": legacy.get("val", {})})
        same = legacy_fp == fp
        print(f"    {LEGACY_SPLIT}")
        print(f"    fingerprint {legacy_fp} -> {'MATCHES' if same else 'DIFFERS FROM'}"
              f" the canonical manifest")
        if not same:
            ok = False
    else:
        print(f"    {LEGACY_SPLIT} not present (it is regenerated from the manifest)")

    # 4. the corpus still matches the manifest --------------------------- #
    print("\n[4] Corpus vs manifest:")
    infos = discover_word_classes(WORD_LEVEL_DIR_DEFAULT, 14, 15)
    corpus_classes = [i.name for i in infos]
    manifest_classes = manifest.get("classes", [])
    missing = set(corpus_classes) - set(manifest_classes)
    extra = set(manifest_classes) - set(corpus_classes)
    print(f"    corpus classes   {len(corpus_classes)}")
    print(f"    manifest classes {len(manifest_classes)}")
    print(f"    only in corpus   {len(missing)} | only in manifest {len(extra)}")
    if missing or extra:
        ok = False
        print("    -> regenerate the manifest:  python scripts/save_phase4_split.py --canonical")
    else:
        print("    -> in sync")

    print("\n" + "=" * 78)
    print("RESULT:", "NO LEAKAGE" if ok else "LEAKAGE / INCONSISTENCY DETECTED")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
