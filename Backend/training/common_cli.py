"""Shared CLI for the word-level phase trainers.

The phase trainers used to ignore ``sys.argv`` completely: ``python
phase2_trainer.py --epochs 1`` silently ran the full 40 epochs, and the
DataLoader was pinned to ``num_workers=0`` so the whole corpus was preloaded
into RAM and every augmentation ran in the main process (one core, 100% load).

This module gives all of them the same options:

    --epochs       how many epochs to run
    --batch_size   the batch size
    --num_workers  how many DataLoader workers load and augment the clips in
                   parallel (2-3 is the sweet spot: the loading AND the
                   augmentation leave the main process, so the GPU is fed
                   instead of waiting for the CPU)
    --out_dir      where the checkpoints go (so a short benchmark run does not
                   overwrite the real ones)
    --data_dir     the word corpus
"""
from __future__ import annotations

import argparse
import os


def add_common_args(parser: argparse.ArgumentParser, epochs: int, batch_size: int,
                     data_dir: str, out_dir: str) -> argparse.ArgumentParser:
    parser.add_argument("--epochs", type=int, default=epochs,
                        help=f"Number of epochs (default: {epochs})")
    parser.add_argument("--batch_size", type=int, default=batch_size,
                        help=f"Batch size (default: {batch_size})")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers: they load AND augment the clips in "
                             "parallel, so the CPU stops being the bottleneck "
                             "(default: 2). Use 0 on a machine with little RAM.")
    parser.add_argument("--data_dir", type=str, default=data_dir,
                        help="The word-level tensor corpus")
    parser.add_argument("--out_dir", type=str, default=out_dir,
                        help="Where the checkpoints, history and plots are written")
    return parser


def make_out_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
