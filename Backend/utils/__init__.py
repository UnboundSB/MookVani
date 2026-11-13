"""Shared utilities: dataloaders, pose augmentation, gloss mapping and metrics."""

from .dataloader_word import (
    MAX_SAMPLES_PER_CLASS,
    MIN_SAMPLES_PER_CLASS,
    NOTHING_CLASS_NAME,
    WORD_LEVEL_DIR_DEFAULT,
    discover_word_classes,
    get_word_classes,
    get_word_dataloaders,
    load_word_class_metadata,
    save_word_class_metadata,
)
from .dataloader_sentence import (
    SENTENCE_DIR_DEFAULT,
    get_sentence_dataloaders,
    load_sentence_index,
)
from .pose_augmentation import (
    AugmentConfig,
    DEFAULT_AUGMENT,
    augment,
    augment_pose_sequence,
    harmonize_features,
    mirror_pose,
    shift_pose,
    tilt_pose,
    time_stretch,
)
from .checkpoint_utils import extract_state_dict, load_flexible
from .ctc_decoder import ctc_prefix_beam_search, decode_with_lm, batch_decode_with_lm
from .plotting import write_report, plot_confusion_matrix, plot_training_curves
from .word_lm import WordLM, build_word_lm_from_index
from .word_distribution import collapse_metrics, mixed_distribution, print_source_report
from .splits import audit_split, load_canonical_split, split_fingerprint
from .gloss_utils import resolve_gloss_sequence, load_sentence_to_words
