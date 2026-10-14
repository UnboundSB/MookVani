# Agent Guidelines and Project Rules

## 1. Directory Structure Maintenance (`structure.md`)
- In every folder and non-ignored subdirectory across the repository, create and maintain an up-to-date `structure.md` file.
- Exclude directories that do not require it: datasets (e.g., `Backend/data/`), virtual environments (`Backend/venv/`, `venv/`), caches (`__pycache__/`, `.ipynb_checkpoints/`), IDE files (`.vscode/`), and other gitignored folders.
- Each `structure.md` must include:
  1. A concise overview of the directory purpose (1-2 lines).
  2. **Files**: The direct files in the current directory with a short 1-2 line description of each.
  3. **Immediate Subdirectories**: The immediate subfolders with a short 1-2 line description of their roles.

## 2. Check Existing Files Before Creating or Editing
- Whenever looking up functionality or about to create/edit a file, check if an appropriate file already exists in the proper directory.
- Always consult the relevant `structure.md` file(s) first to verify existing implementations.
- Modify and extend existing files instead of creating new, duplicate, or scratch files.
- Whenever files or subfolders are added, removed, or reorganized, immediately update the corresponding `structure.md` file(s).

## 3. Protected Files (NEVER MODIFY OR DELETE)
- `yap_conformer.py` is the MOST IMPORTANT file in the repository. Do NOT modify or delete it under any circumstances unless explicitly instructed by the user.
- `.gitignore`, `README.md`, and any files/tasks that extract MediaPipe coordinates from videos (including `*.task` landmark models like `face_landmarker.task`, `hand_landmarker.task`, `holistic_landmarker.task`, `pose_landmarker.task`, and extraction scripts like `extraction.py`, `extract_features.py`) are critical and must NEVER be deleted.

## 4. Repository Cleanliness & Anti-Clutter
- Do not create scratch files (`scratch_*.py`), backup copies (`*.bak`, `*_backup.py`), or duplicate files.
- Keep `Backend/models/` clean, maintaining Python source files and designated model directories rather than loose duplicate checkpoints or temporary non-Python dumps.
- Store generated plots in their designated visualization directories (e.g., `Backend/eda/plots/` or model-specific plot subdirectories) rather than scattering images across root or training directories.
