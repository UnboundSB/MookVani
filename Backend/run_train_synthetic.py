import os
import sys
import time

# Support UTF-8 output if possible
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure Backend root is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from training.train_synthetic_sentences import train_synthetic_sentences
from training.evaluate_synthetic_sentences import evaluate

def main():
    print("=" * 70)
    print("MOOKVANI SYNTHETIC SENTENCE PRODUCTION TRAINING PIPELINE")
    print("=" * 70)
    print(" Architecture:      ISLSentenceReconformer (BiGRU + Transformer + CNN)")
    print(" Pretrained Base:   reconformer_word_135_best.pth (44.36% Word Accuracy)")
    print(" Vocabulary:        135 balanced classes (>=15 samples, capped at 15 in memory)")
    print(" Sequence Range:    1 to 3 words per sentence (High-Density Recognition)")
    print(" Co-articulation:   Smooth 3-5 frame motion blending between consecutive words")
    print(" Stage Epochs:      10 epochs (Optimal Early Stopping before gradient drift)")
    print(" Differential LR:   2.0e-4 (RNN) / 2.0e-5 (Backbone)")
    print(" Emission Penalty:  0.50 (Strict Anti-Blank Penalty for active emissions)")
    print(" Batch Size:        4 (with periodic gc.collect and cuda cache clear)")
    print("=" * 70 + "\n")
    
    # Configure CLI arguments: Fresh start from pretrained word model
    sys.argv = [
        'train_synthetic_sentences.py',
        '--epochs', '10',
        '--min_words', '1',
        '--max_words', '3',
        '--lr', '2.0e-4',
        '--emission_penalty', '0.50',
        '--batch_size', '4'
    ]
    
    start_time = time.time()
    
    # Run Production Training
    train_synthetic_sentences()
    
    elapsed = time.time() - start_time
    print(f"\nProduction Training completed in {elapsed / 60:.2f} minutes!")
    print("\n" + "=" * 70)
    print("RUNNING AUTOMATED EVALUATION WITH TEMPORAL DURATION FILTER")
    print("=" * 70)
    
    # Run Evaluation on 1 to 3 word test sentences with duration filtering
    try:
        evaluate(num_samples=25, min_words=1, max_words=3)
    except Exception as e:
        print(f"Evaluation encountered an issue: {e}")

if __name__ == "__main__":
    main()
