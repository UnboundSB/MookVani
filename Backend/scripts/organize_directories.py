import os
import shutil

BACKEND_DIR = r"d:\MookVani\Backend"
EDA_PLOTS_DIR = os.path.join(BACKEND_DIR, "eda", "plots")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")

# 1. Clean non-EDA plots from eda/plots
non_eda_files = [
    "ctc_metrics.png",
    "loss_and_wer.png",
    "reconformer_confusion_matrix.png",
    "reconformer_confusion_matrix_final.png",
    "reconformer_confusion_matrix_updated.png",
    "reconformer_metrics.png",
    "synthetic_training_plots.png",
    "word_135_confusion_matrix.png",
    "word_135_metrics.png"
]

for f in non_eda_files:
    p = os.path.join(EDA_PLOTS_DIR, f)
    if os.path.exists(p):
        try:
            os.remove(p)
            print(f"Removed non-EDA plot from eda/plots: {f}")
        except Exception as e:
            print(f"Could not remove {p}: {e}")

# 2. Setup separate directories for each model with its respective plots
# Directory for 229 Balanced Word Model
dir_229 = os.path.join(MODELS_DIR, "word_reconformer_229")
os.makedirs(dir_229, exist_ok=True)

# Directory for 135 Word Model
dir_135 = os.path.join(MODELS_DIR, "word_reconformer_135")
os.makedirs(dir_135, exist_ok=True)

# Directory for 264 Final Word Model
dir_264 = os.path.join(MODELS_DIR, "word_reconformer_264")
os.makedirs(dir_264, exist_ok=True)

# Directory for CTC Sentence Model
dir_ctc = os.path.join(MODELS_DIR, "sentence_ctc")
os.makedirs(dir_ctc, exist_ok=True)

# Move 135 files into word_reconformer_135
files_135 = [
    ("word_level/reconformer_word_135_best.pth", "reconformer_word_135_best.pth"),
    ("word_level/reconformer_word_135_final.pth", "reconformer_word_135_final.pth"),
    ("word_level/classes_135.json", "classes_135.json"),
    ("word_level/plots_135/word_135_confusion_matrix.png", "word_135_confusion_matrix.png"),
    ("word_level/plots_135/word_135_metrics.png", "word_135_metrics.png"),
]
for src_rel, dst_name in files_135:
    src_path = os.path.join(MODELS_DIR, src_rel)
    dst_path = os.path.join(dir_135, dst_name)
    if os.path.exists(src_path):
        try:
            shutil.copy2(src_path, dst_path)
            print(f"Organized 135 file -> {dst_name}")
        except Exception as e:
            print(f"Error copying {src_path}: {e}")

# Move 264 final files into word_reconformer_264
files_264 = [
    ("word_level/reconformer_word_model_final.pth", "reconformer_word_model_final.pth"),
    ("word_level/reconformer_confusion_matrix_final.png", "reconformer_confusion_matrix_final.png"),
    ("word_level/reconformer_classification_report_final.txt", "reconformer_classification_report_final.txt"),
]
for src_rel, dst_name in files_264:
    src_path = os.path.join(MODELS_DIR, src_rel)
    dst_path = os.path.join(dir_264, dst_name)
    if os.path.exists(src_path):
        try:
            shutil.copy2(src_path, dst_path)
            print(f"Organized 264 file -> {dst_name}")
        except Exception as e:
            print(f"Error copying {src_path}: {e}")

# Move CTC sentence files into sentence_ctc
files_ctc = [
    ("sentence_level/isl_sentence_ctc.pth", "isl_sentence_ctc.pth"),
]
for src_rel, dst_name in files_ctc:
    src_path = os.path.join(MODELS_DIR, src_rel)
    dst_path = os.path.join(dir_ctc, dst_name)
    if os.path.exists(src_path):
        try:
            shutil.copy2(src_path, dst_path)
            print(f"Organized CTC file -> {dst_name}")
        except Exception as e:
            print(f"Error copying {src_path}: {e}")

# Move 229 files into word_reconformer_229
files_229 = [
    ("word_level/classes_balanced.json", "classes_balanced.json"),
    ("word_level/history_word.json", "history_word.json"),
]
for src_rel, dst_name in files_229:
    src_path = os.path.join(MODELS_DIR, src_rel)
    dst_path = os.path.join(dir_229, dst_name)
    if os.path.exists(src_path):
        try:
            shutil.copy2(src_path, dst_path)
            print(f"Organized 229 file -> {dst_name}")
        except Exception as e:
            print(f"Error copying {src_path}: {e}")

print("Directory reorganization complete!")
