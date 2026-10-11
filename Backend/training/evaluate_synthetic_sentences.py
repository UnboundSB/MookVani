import os
import sys
import json
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.isl_sentence_model import ISLSentenceReconformer
from training.train_synthetic_sentences import (
    SyntheticSentenceDataset,
    collate_fn_sentence,
    ctc_decode,
    get_targets,
    compute_wer,
    compute_f1_prec_rec
)

def evaluate(num_samples=20, max_words=6):
    MODEL_DIR = r"d:\MookVani\Backend\models\islsentence_synthetic"
    DATA_DIR = r"d:\MookVani\Backend\data\tensors_include50_word_level"
    CLASSES_PATH = os.path.join(MODEL_DIR, "classes.json")
    BEST_WEIGHTS_PATH = os.path.join(MODEL_DIR, "reconformer_synthetic_sentence_best.pth")
    
    if not os.path.exists(CLASSES_PATH) or not os.path.exists(BEST_WEIGHTS_PATH):
        print(f"Error: Required model files not found in {MODEL_DIR}")
        return
        
    with open(CLASSES_PATH, "r") as f:
        meta = json.load(f)
        classes = meta["classes"]
        num_classes = meta["num_classes"]
        ctc_blank_id = meta["ctc_blank_id"]
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating Reconformer CTC on {device} (Classes: {num_classes}, Blank ID: {ctc_blank_id})")
    
    model = ISLSentenceReconformer(num_classes=num_classes, d_model=128, lstm_layers=1, dropout=0.5, use_rnn=True).to(device)
    model.load_state_dict(torch.load(BEST_WEIGHTS_PATH, map_location=device, weights_only=True))
    model.eval()
    
    dataset = SyntheticSentenceDataset(DATA_DIR, classes, samples_per_epoch=num_samples, max_words=max_words, is_train=False)
    
    total_wer = 0.0
    exact_matches = 0
    total_f1 = 0.0
    
    print("\n" + "=" * 80)
    print(f"{'SAMPLE':<6} | {'GROUND TRUTH':<35} | {'PREDICTION':<35}")
    print("=" * 80)
    
    for idx in range(num_samples):
        full_tensor, length, labels = dataset[idx]
        x = full_tensor.unsqueeze(0).to(device)
        lengths = torch.tensor([length]).to(device)
        
        with torch.no_grad():
            log_probs, pooled_lengths = model(x, lengths)
            log_probs = log_probs.transpose(0, 1) # [T, B, C]
            decoded_indices = ctc_decode(log_probs, pooled_lengths, blank=ctc_blank_id)[0]
            
        ref_indices = labels.numpy().tolist()
        
        # Filter out NONE (silence) for word-level evaluation
        none_idx = classes.index("NONE") if "NONE" in classes else -1
        ref_words = [classes[i] for i in ref_indices if i != none_idx and i < len(classes)]
        hyp_words = [classes[i] for i in decoded_indices if i != none_idx and i < len(classes)]
        
        ref_str = " ".join(ref_words) if ref_words else "[SILENCE]"
        hyp_str = " ".join(hyp_words) if hyp_words else "[SILENCE]"
        
        wer = compute_wer(ref_words, hyp_words)
        f1, _, _ = compute_f1_prec_rec(ref_words, hyp_words)
        is_match = (ref_words == hyp_words)
        
        total_wer += wer
        total_f1 += f1
        if is_match:
            exact_matches += 1
            
        # Display first 15 samples
        if idx < 15:
            ref_display = (ref_str[:32] + "..") if len(ref_str) > 34 else ref_str
            hyp_display = (hyp_str[:32] + "..") if len(hyp_str) > 34 else hyp_str
            match_sym = "✅" if is_match else "  "
            print(f"{idx+1:<6} | {ref_display:<35} | {hyp_display:<35} {match_sym}")
            
    avg_wer = total_wer / num_samples
    avg_f1 = total_f1 / num_samples
    exact_acc = (exact_matches / num_samples) * 100.0
    
    print("=" * 80)
    print(f"Evaluation Results across {num_samples} Synthetic Sentences:")
    print(f" • Average Word Error Rate (WER): {avg_wer:.4f}")
    print(f" • Average Sequence F1:            {avg_f1:.4f}")
    print(f" • Exact Sentence Match:          {exact_acc:.1f}% ({exact_matches}/{num_samples})")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    evaluate(num_samples=25, max_words=5)
