"""
Central configuration for YAP Segmentor v5.

Single source of truth for paths, hardware, model, training and regularization
hyper-parameters. Import ``CFG`` from here instead of scattering constants.
"""

from pathlib import Path
import torch


class Config:
    # ------------------------------------------------------------------ paths
    SEG_CSV = Path(r"D:\MookVani\backend\segmentor\data_builder\segmentation_clean.csv")
    SENTENCES_CSV = Path(r"D:\MookVani\backend\segmentor\data_builder\sentences.csv")
    WORD_INVENTORY = Path(r"D:\MookVani\backend\segmentor\data_builder\word_inventory.json")
    UNLABELED_DIR = Path(r"D:\MookVani\backend\data\ISL_CSLRT_Corpus\Videos_Sentence_Level")

    CHECKPOINT_DIR = Path(r"D:\MookVani\Backend\models\yap_segmentor_v5")
    FEATURES_CACHE_DIR = Path(r"D:\MookVani\backend\segmentor\features_cache")
    PRETRAINED_CKPT = Path(r"D:\MookVani\Backend\models\yap_segmentor\yap_segmentor_final.pth")

    # create output dirs on import
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ model / data
    FPS = 25
    MAX_LEN = 512
    NUM_JOINTS = 42  # 2 hands x 21 joints

    # ------------------------------------------------------------------ training
    EPOCHS = 50
    LABELED_BATCH_SIZE = 8
    UNLABELED_BATCH_SIZE = 8
    GRAD_ACCUM_STEPS = 2            # effective batch = 16
    LR = 1e-4
    WEIGHT_DECAY = 1e-3
    GRAD_CLIP_NORM = 1.0

    LR_SCHEDULER = "plateau"        # "cosine" | "plateau"
    PLATEAU_PATIENCE = 5
    PLATEAU_FACTOR = 0.5
    MIN_LR = 1e-6

    DROPOUT = 0.1
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2

    # ------------------------------------------------------------------ semi-supervised (Mean Teacher)
    EMA_DECAY = 0.999
    PSEUDO_LABEL_EVERY = 3
    PSEUDO_PRESENCE_THRESH = 0.95
    PSEUDO_END_THRESH = 0.8
    PSEUDO_MIN_CONF = 0.7
    CONSISTENCY_WEIGHT = 0.1
    CONSISTENCY_RAMP_EPOCHS = 10

    # ------------------------------------------------------------------ losses
    PRESENCE_POS_WEIGHT = 2.0
    END_POS_WEIGHT = 10.0
    LABEL_SMOOTHING = 0.1

    BOUNDARY_NMS_WEIGHT = 0.01
    BOUNDARY_MIN_DIST = 5

    USE_AMP = True
    SAVE_EVERY = 3
    KEEP_BEST_N = 3

    # ------------------------------------------------------------------ early stopping
    EARLY_STOP_PATIENCE = 8
    MIN_DELTA = 1e-4

    # ------------------------------------------------------------------ split
    VAL_SPLIT = 0.2
    SEED = 42


CFG = Config()
