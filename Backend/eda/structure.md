# Directory Structure: Backend/eda

Exploratory Data Analysis (EDA) module for dataset inspection, format verification, and visual distribution generation.

## Files
- `check_dataset.py`: Validates dataset folder organization and directory integrity.
- `check_file_type.py`: Verifies file types and video encoding formats across dataset sources.
- `check_progress.py`: Monitors progress for batch extraction and feature processing jobs.
- `check_word_collision.py`: Checks for overlapping vocabulary and collisions across gloss sets.
- `extract_dataset.py`: Extracts raw video and metadata archives for the primary dataset.
- `extract_isign_only.py`: Unpacks and filters subsets specifically from the iSign dataset.
- `generate_isl_cslrt_eda.py`: Standalone EDA script generating comprehensive distribution plots for ISL-CSLRT (word and sentence levels).
- `inspect_eda.py`: Console-based data inspector printing high-level dataset metrics.
- `sentence_class_frequency_eda.py`: Analyzes distribution and frequencies of trained vocabulary classes in continuous sentences.
- `sentence_class_frequencies.json`: Stored JSON frequency metrics and vocabulary coverage data.
- `structure.md`: Directory structure documentation for eda.

## Immediate Subdirectories
- `plots/`: Output directory containing generated EDA charts, distribution graphs, and overview dashboards (e.g., `sentence_active_words_frequency.png`, `sentence_all_262_class_frequencies.png`, `sentence_word_frequencies.png`).
