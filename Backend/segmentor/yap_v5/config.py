"""
Configuration for YAP Segmentor v5
"""

import torch
from pathlib import Path

class Config:
    # Paths
    SEG_CSV = Path(r"D:\MookVani\backend\segmentor\data_builder\segmentation_clean.csv")
    SENTENCES_CSV = Path(r"D:\MookVani\backend\segmentor\data_builder\sentences.csv")
    WORD_INVENTORY = Path(r"D:\MookVani\backend\segmentor\data_builder\word_inventory.json")
    UNLABELED_DIR = Path(r"D:\MookVani\backend\data\ISL_CSLRT_Corpus\Videos_Sentence_Level")
    
    CHECKPOINT_DIR = Path(r"D:\MookVani\Backend\models\yap_segmentor_v5")
    FEATURES_CACHE_DIR = Path(r"D:\MookVani\backend\segmentor\features_cache")
    
    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model
    FPS = 25
    MAX_LEN = 512
    NUM_JOINTS = 42  # 2 hands × 21
    
    # Training
    EPOCHS = 50
    LABELED_BATCH_SIZE = 8
    UNLABELED_BATCH_SIZE = 8
    GRAD_ACCUM_STEPS = 2
    LR = 1e-4
    WEIGHT_DECAY = 1e-3
    GRAD_CLIP_NORM = 1.0
    
    # Optimization
    LR_SCHEDULER = "plateau"
    PLATEAU_PATIENCE = 5
    PLATEAU_FACTOR = 0.5
    MIN_LR = 1e-6
    
    # Regularization
    WEIGHT_DECAY = 1e-3
    DROPOUT = 0.1
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2
    
    # Semi-supervised
    EMA_DECAY = 0.999
    PSEUDO_LABEL_EVERY = 3
    PSEUDO_PRESENCE_THRESH = 0.95
    PSEUDO_END_THRESH = 0.8
    PSEUDO_MIN_CONF = 0.7
    CONSISTENCY_WEIGHT = 0.1
    CONSISTENCY_RAMP_EPOCHS = 10
    
    # Losses
    PRESENCE_POS_WEIGHT = 2.0
    END_POS_WEIGHT = 10.0
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2
    
    # Boundary loss
    BOUNDARY_NMS_WEIGHT = 0.01
    BOUNDARY_MIN_DIST = 5
    
    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LEN = 512
    FPS = 25
    USE_AMP = True
    GRAD_ACCUM_STEPS = 2
    
    # Checkpointing
    SAVE_EVERY = 3
    KEEP_BEST_N = 3
    
    # Early stopping
    EARLY_STOP_PATIENCE = 8
    MIN_DELTA = 1e-4
    
    # Split
    VAL_SPLIT = 0.2
    SEED = 42
    
    # Paths
    SEG_CSV = Path(r"D:\MookVani\backend\segmentor\data_builder\segmentation_clean.csv")
    SENTENCES_CSV = Path(r"D:\MookVani\backend\segmentor\data_builder\sentences.csv")
    WORD_INVENTORY = Path(r"D:\MookVani\backend\segmentor\data_builder\word_inventory.json")
    UNLABELED_DIR = Path(r"D:\MookVani\backend\data\ISL_CSLRT_Corpus\Videos_Sentence_Level")
    
    CHECKPOINT_DIR = Path(r"D:\MookVani\Backend\models\yap_segmentor_v5")
    FEATURES_CACHE_DIR = Path(r"D:\MookVani\backend\segmentor\features_cache")


cfg = Config()