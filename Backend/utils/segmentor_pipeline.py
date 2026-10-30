"""
Segmentor training pipeline orchestrator.

Links the segmentor stages dynamically so only the relevant stuff runs:

    stage 0 — build_word_dataset   (segmentor/data/<word>/<take>/word.pt + processed.pt)
    stage 1 — noise filtration     (yap_noise_robust.pth)
    stage 2A — word-entity match   (yap_phaseA_best.pth + word_head)
    stage 2B — synthetic segmentation (yap_seg_best.pth)
    stage 3 — real-sentence fine-tune (delegates to ../training/train_real_sentences.py)

Usage::

    from utils.segmentor_pipeline import SegmentorPipeline
    pipe = SegmentorPipeline(backend_dir=Path("Backend"))
    pipe.run("stage1", epochs=30, batch_size=4, ...)
    pipe.run("stage2a", epochs=60, ...)
    pipe.run_all(preset="fast")   # stage0 -> stage1 -> stage2a -> stage2b
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_SEGMENTOR_DIR = _BACKEND_DIR / "segmentor"
for _p in (str(_BACKEND_DIR), str(_SEGMENTOR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.segmentor_utils import (  # noqa: E402
    LAYOUTS, SegmentorModelCfg, DEFAULT_CLASSES_PATH, DEFAULT_WORD_CORPUS,
    DEFAULT_SEGMENTOR_DATA, DEFAULT_STAGE1_OUT, DEFAULT_STAGE2_OUT,
    create_yap_model, load_yap_weights,
    discover_segmentor_words, segmentor_take_items,
    SegmentorWordClip, segmentor_collate,
    save_segmentor_ckpt, load_segmentor_ckpt,
    yap_losses, conformer_ce_loss, write_segmentor_report,
)


@dataclass
class PipelineStage:
    name: str
    enabled: bool = False
    depends_on: List[str] = field(default_factory=list)
    fn: Optional[Callable] = None


class SegmentorPipeline:
    """
    Dynamically linked segmentor training pipeline.

    Each stage is a thin wrapper around the corresponding training script.
    Stages declare their dependencies; run(stage) automatically runs
    prerequisites first if they haven't been completed.
    """

    def __init__(
        self,
        backend_dir: Path | str = _BACKEND_DIR,
        segmentor_dir: Path | str = _SEGMENTOR_DIR,
        classes_path: Path | str = DEFAULT_CLASSES_PATH,
        word_corpus: Path | str = DEFAULT_WORD_CORPUS,
        segmentor_data: Path | str = DEFAULT_SEGMENTOR_DATA,
        stage1_out: Path | str = DEFAULT_STAGE1_OUT,
        stage2_out: Path | str = DEFAULT_STAGE2_OUT,
        device: Optional[str] = None,
    ):
        self.backend_dir = Path(backend_dir)
        self.segmentor_dir = Path(segmentor_dir)
        self.classes_path = Path(classes_path)
        self.word_corpus = Path(word_corpus)
        self.segmentor_data = Path(segmentor_data)
        self.stage1_out = Path(stage1_out)
        self.stage2_out = Path(stage2_out)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        with open(self.classes_path, encoding="utf-8") as f:
            self.classes = json.load(f)
        self.num_classes = len(self.classes)

        # Completion markers (files that prove a stage finished)
        self._markers: Dict[str, Path] = {
            "stage0": self.segmentor_data / "mapping.json",
            "stage1": self.stage1_out / "yap_noise_robust.pth",
            "stage2a": self.stage2_out / "yap_phaseA_best.pth",
            "stage2b": self.stage2_out / "yap_seg_best.pth",
        }

    # ------------------------------------------------------------------ #
    # Completion checks
    # ------------------------------------------------------------------ #
    def is_done(self, stage: str) -> bool:
        marker = self._markers.get(stage)
        return marker is not None and marker.is_file()

    def mark_done(self, stage: str) -> None:
        """No-op: the marker files themselves are the proof of completion."""

    # ------------------------------------------------------------------ #
    # Stage runners (each delegates to the actual training script)
    # ------------------------------------------------------------------ #
    def run_stage0_build_word_dataset(
        self,
        phase: int = 3,
        max_per_class: int = 22,
        max_len: int = 900,
        fps: float = 30.0,
        num_workers: int = 4,
        max_gap_frames: int = 15,
        layout: str = "corpus",
        d_model: int = 128,
        n_mem_layers: int = 3,
        summary_channels: int = 32,
        hippo_order: int = 16,
        entity_dim: int = 256,
        heads: int = 4,
        layers: int = 4,
        init: Optional[Path | str] = None,
        overwrite: bool = False,
    ) -> None:
        """Stage 0: build segmentor/data/<word>/<take>/ (word.pt + processed.pt)."""
        if self.is_done("stage0") and not overwrite:
            print(f"[PIPELINE] stage0 already done ({self._markers['stage0']}); skipping")
            return

        from build_word_dataset import build_dataset
        from types import SimpleNamespace

        args = SimpleNamespace(
            data_src=str(self.word_corpus),
            classes=str(self.classes_path),
            init=str(init) if init else str(self.stage1_out / "yap_noise_robust.pth"),
            out_root=str(self.segmentor_data),
            layout=layout,
            fps=fps,
            max_len=max_len,
            max_per_class=max_per_class,
            phase=phase,
            num_workers=num_workers,
            max_gap_frames=max_gap_frames,
            d_model=d_model,
            n_mem_layers=n_mem_layers,
            summary_channels=summary_channels,
            hippo_order=hippo_order,
            entity_dim=entity_dim,
            heads=heads,
            layers=layers,
            overwrite=overwrite,
        )
        build_dataset(args)

    def run_stage1_noise_filter(
        self,
        epochs: int = 30,
        batch_size: int = 4,
        grad_accum: int = 2,
        lr: float = 3e-4,
        out_dir: Optional[Path | str] = None,
        data_dirs: Optional[List[str]] = None,
        init: Optional[Path | str] = None,
        resume: bool = False,
        layout: str = "corpus",
        d_model: int = 128,
        n_mem_layers: int = 3,
        summary_channels: int = 32,
        hippo_order: int = 16,
        entity_dim: int = 256,
        t0: int = 5,
        t_mult: int = 2,
        max_len: int = 400,
        max_per_class: Optional[int] = None,
        seed: int = 42,
        num_workers: int = 2,
        jitter_std: float = 0.015,
        noise_scale_min: float = 0.0,
        noise_scale_max: float = 1.0,
        noise_prob: float = 0.9,
        aug: int = 1,
        aug_strength: float = 1.0,
        ema_decay: float = 0.99,
        cls_w: float = 1.0,
        ent_w: float = 0.5,
        blank_w: float = 1.0,
        blank_floor: float = 0.15,
        div_w: float = 0.5,
        overwrite: bool = False,
    ) -> None:
        """Stage 1: noise-robustness training -> yap_noise_robust.pth."""
        out_dir = Path(out_dir) if out_dir else self.stage1_out
        ckpt = out_dir / "yap_noise_robust.pth"
        if ckpt.is_file() and not overwrite:
            print(f"[PIPELINE] stage1 already done ({ckpt}); skipping")
            return

        from train_noise_filter import train as noise_train
        from types import SimpleNamespace

        args = SimpleNamespace(
            data_dirs=data_dirs or [str(self.word_corpus)],
            classes=str(self.classes_path),
            out_dir=str(out_dir),
            init=str(init) if init else None,
            resume=resume,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum=grad_accum,
            layout=layout,
            num_workers=num_workers,
            jitter_std=jitter_std,
            noise_scale_min=noise_scale_min,
            noise_scale_max=noise_scale_max,
            noise_prob=noise_prob,
            aug=aug,
            aug_strength=aug_strength,
            lr=lr,
            ema_decay=ema_decay,
            cls_w=cls_w,
            ent_w=ent_w,
            blank_w=blank_w,
            blank_floor=blank_floor,
            div_w=div_w,
            t0=t0,
            t_mult=t_mult,
            max_len=max_len,
            max_per_class=max_per_class,
            seed=seed,
            d_model=d_model,
            n_mem_layers=n_mem_layers,
            summary_channels=summary_channels,
            hippo_order=hippo_order,
            entity_dim=entity_dim,
        )
        noise_train(args)

    def run_stage2a_word_entity(
        self,
        epochs: int = 60,
        batch_size: int = 4,
        lr: float = 1e-4,
        out_dir: Optional[Path | str] = None,
        dataset_root: Optional[Path | str] = None,
        init: Optional[Path | str] = None,
        overwrite: bool = False,
        layout: str = "corpus",
        d_model: int = 128,
        n_mem_layers: int = 3,
        summary_channels: int = 32,
        hippo_order: int = 16,
        entity_dim: int = 256,
        max_per_class: int = 22,
        max_words: int = 5,
        max_len: int = 900,
        fps: float = 30.0,
        seed: int = 42,
        cpu: bool = False,
    ) -> None:
        """Stage 2A: word-entity matching -> yap_phaseA_best.pth + word_head."""
        out_dir = Path(out_dir) if out_dir else self.stage2_out
        ckpt = out_dir / "yap_phaseA_best.pth"
        if ckpt.is_file() and not overwrite:
            print(f"[PIPELINE] stage2a already done ({ckpt}); skipping")
            return

        _run_script(
            str(self.segmentor_dir / "train_segmentor.py"),
            [
                "--phase", "A",
                "--dataset-root", str(dataset_root) if dataset_root else str(self.segmentor_data),
                "--classes", str(self.classes_path),
                "--init", str(init) if init else str(self.stage1_out / "yap_noise_robust.pth"),
                "--out-dir", str(out_dir),
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--train-sentences", "4000",
                "--val-sentences", "400",
                "--max-per-class", str(max_per_class),
                "--max-words", str(max_words),
                "--max-len", str(max_len),
                "--fps", str(fps),
                "--lr", str(lr),
                "--seed", str(seed),
                "--d-model", str(d_model),
                "--n-mem-layers", str(n_mem_layers),
                "--summary-channels", str(summary_channels),
                "--hippo-order", str(hippo_order),
                "--entity-dim", str(entity_dim),
            ] + (["--cpu"] if cpu else [])
            + (["--overwrite"] if overwrite else []),
        )

    def run_stage2b_synthetic_segmentation(
        self,
        epochs: int = 60,
        batch_size: int = 4,
        lr: float = 1e-4,
        out_dir: Optional[Path | str] = None,
        dataset_root: Optional[Path | str] = None,
        init: Optional[Path | str] = None,
        overwrite: bool = False,
        layout: str = "corpus",
        d_model: int = 128,
        n_mem_layers: int = 3,
        summary_channels: int = 32,
        hippo_order: int = 16,
        entity_dim: int = 256,
        max_per_class: int = 22,
        max_words: int = 5,
        max_len: int = 900,
        fps: float = 30.0,
        seed: int = 42,
        cpu: bool = False,
    ) -> None:
        """Stage 2B: synthetic sentence segmentation + CTC -> yap_seg_best.pth."""
        out_dir = Path(out_dir) if out_dir else self.stage2_out
        ckpt = out_dir / "yap_seg_best.pth"
        if ckpt.is_file() and not overwrite:
            print(f"[PIPELINE] stage2b already done ({ckpt}); skipping")
            return

        _run_script(
            str(self.segmentor_dir / "train_segmentor.py"),
            [
                "--phase", "B",
                "--dataset-root", str(dataset_root) if dataset_root else str(self.segmentor_data),
                "--classes", str(self.classes_path),
                "--init", str(init) if init else str(self.stage2_out / "yap_phaseA_best.pth"),
                "--out-dir", str(out_dir) if out_dir else str(self.stage2_out),
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--train-sentences", "4000",
                "--val-sentences", "400",
                "--max-per-class", str(max_per_class),
                "--max-words", str(max_words),
                "--max-len", str(max_len),
                "--fps", str(fps),
                "--lr", str(lr),
                "--seed", str(seed),
                "--d-model", str(d_model),
                "--n-mem-layers", str(n_mem_layers),
                "--summary-channels", str(summary_channels),
                "--hippo-order", str(hippo_order),
                "--entity-dim", str(entity_dim),
            ] + (["--cpu"] if cpu else [])
            + (["--overwrite"] if overwrite else []),
        )

    def run_stage3_real_sentence(
        self,
        init: Optional[Path | str] = None,
        **kwargs,
    ) -> None:
        """Stage 3: real-sentence fine-tune (delegates to ../training/train_real_sentences.py)."""
        script = self.backend_dir / "training" / "train_real_sentences.py"
        if not script.is_file():
            print(f"[PIPELINE] stage3 script not found: {script}")
            return
        cmd = [sys.executable, str(script)]
        if init:
            cmd += ["--init", str(init)]
        for k, v in kwargs.items():
            cmd += [f"--{k}", str(v)]
        subprocess.run(cmd, check=True)

    # ------------------------------------------------------------------ #
    # Dynamic dispatch
    # ------------------------------------------------------------------ #
    def run(self, stage: str, **kwargs) -> None:
        """Run a single stage, auto-running prerequisites if missing."""
        deps = {
            "stage0": [],
            "stage1": ["stage0"],
            "stage2a": ["stage1"],
            "stage2b": ["stage2a"],
            "stage3": ["stage2b"],
        }
        for dep in deps.get(stage, []):
            if not self.is_done(dep):
                print(f"[PIPELINE] {stage} needs {dep}; running {dep} first")
                self.run(dep, **kwargs)

        runner = getattr(self, f"run_{stage}", None)
        if runner is None:
            raise ValueError(f"Unknown stage: {stage}")
        print(f"[PIPELINE] === RUNNING {stage} ===")
        runner(**kwargs)
        print(f"[PIPELINE] === {stage} DONE ===")

    def run_all(
        self,
        preset: str = "fast",
        **kwargs,
    ) -> None:
        """
        Run the full pipeline in order.

        preset="fast"   : stage0 -> stage1 -> stage2a -> stage2b
        preset="accuracy": same + stage3
        """
        order = ["stage0", "stage1", "stage2a", "stage2b"]
        if preset == "accuracy":
            order.append("stage3")
        for stage in order:
            self.run(stage, **kwargs)


def _run_script(script_path: str, extra_args: List[str]) -> None:
    """Launch a training script as a subprocess so argparse stays intact."""
    cmd = [sys.executable, script_path] + extra_args
    print(f"[PIPELINE] launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
