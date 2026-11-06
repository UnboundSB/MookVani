"""yapv3.utils — data, losses, model, trainer modules for YAP gesture segmentation."""
from __future__ import annotations

from .data import (AugmentationConfig, SentenceClipDataset, build_sentence_dataset,
                   collate_sentence_batch, stitch_random_words)
from .ctc_sentence_data import (RealSentenceCTCDataset, collate_ctc_batch,
                                build_real_sentence_dataset, match_sentences_to_tensors,
                                load_segmentation_marks, build_segmentation_boundary_items,
                                collate_segmentation_batch)
from .losses import (focal_presence_loss, end_bce_loss, layer_conf_loss,
                     soft_count_loss, total_segmentation_loss)
from .ctc_sentence_trainer import (AntiCollapseCTCLoss, greedy_decode, wer_score,
                                   train_epoch, validate, run_ctc_sentences)
from .checker import (CollapseThresholds, check_model_forward, check_history,
                      run_universal_check, save_check_report)
from .synthetic_sentence_data import (SyntheticSentenceDataset,
                                      collate_synthetic_batch,
                                      build_synthetic_dataset)
from .model import load_yap_model, YAPModelWrapper
from .trainer import run_phase3

__all__ = [
    # data
    "AugmentationConfig",
    "SentenceClipDataset",
    "build_sentence_dataset",
    "collate_sentence_batch",
    "stitch_random_words",
    # ctc_sentence_data
    "RealSentenceCTCDataset",
    "collate_ctc_batch",
    "build_real_sentence_dataset",
    "match_sentences_to_tensors",
    "load_segmentation_marks",
    "build_segmentation_boundary_items",
    "collate_segmentation_batch",
    # synthetic_sentence_data
    "SyntheticSentenceDataset",
    "collate_synthetic_batch",
    "build_synthetic_dataset",
    # losses
    "focal_presence_loss",
    "end_bce_loss",
    "layer_conf_loss",
    "soft_count_loss",
    "total_segmentation_loss",
    # ctc_sentence_trainer
    "AntiCollapseCTCLoss",
    "greedy_decode",
    "wer_score",
    "train_epoch",
    "validate",
    "run_ctc_sentences",
    # checker
    "CollapseThresholds",
    "check_model_forward",
    "check_history",
    "run_universal_check",
    "save_check_report",
    # model
    "load_yap_model",
    "YAPModelWrapper",
    # trainer
    "run_phase3",
]
