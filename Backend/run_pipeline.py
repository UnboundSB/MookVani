import os
import subprocess
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"

def run_script(script_path):
    print(f"\n==============================================")
    print(f"Running {script_path}...")
    print(f"==============================================")
    result = subprocess.run([sys.executable, script_path], cwd="d:/MookVani/Backend")
    if result.returncode != 0:
        print(f"Error running {script_path}")
        sys.exit(1)

if __name__ == "__main__":
    scripts = [
        ["scripts/extract_word.py"],
        ["scratch_dedup_confound.py"],
        ["scripts/augment_extract_sentence.py"],
        ["training/train_word.py", "--train_dir", "data/tensors_word_level_163_train", "--val_dir", "data/tensors_word_level_163_val"],
        ["training/train_synthetic.py"]
    ]
    
    for script_args in scripts:
        print(f"\n==============================================")
        print(f"Running {' '.join(script_args)}...")
        print(f"==============================================")
        
        cmd = [sys.executable] + script_args
        result = subprocess.run(cmd, cwd="d:/MookVani/Backend")
        if result.returncode != 0:
            print(f"Error running {' '.join(script_args)}")
            sys.exit(1)
        
    print("\nAll tasks completed successfully!")
