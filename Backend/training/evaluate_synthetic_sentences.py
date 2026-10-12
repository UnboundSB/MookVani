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

def decode_with_min_duration(log_probs, blank_id, none_idx, min_duration=3):
    preds = log_probs.argmax(dim=-1).cpu().numpy()[0]
    runs = []
    curr_tok = preds[0]
    curr_len = 1
    for t in range(1, len(preds)):
        if preds[t] == curr_tok:
            curr_len += 1
        else:
            runs.append((curr_tok, curr_len))
            curr_tok = preds[t]
            curr_len = 1
    runs.append((curr_tok, curr_len))
    
    decoded = []
    for tok, length in runs:
        if tok != blank_id and tok != none_idx:
            if length >= min_duration:
                if len(decoded) == 0 or decoded[-1] != tok:
                    decoded.append(tok)
    return decoded

def evaluate(num_samples=25, min_words=2, max_words=10):
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
        none_idx = classes.index("NONE") if "NONE" in classes else -1
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating Reconformer CTC on {device} (Classes: {num_classes}, Blank ID: {ctc_blank_id})")
    print(f"Sentence Word Count Range: {min_words} to {max_words} words")
    
    model = ISLSentenceReconformer(num_classes=num_classes, d_model=128, lstm_layers=1, dropout=0.3, use_rnn=True).to(device)
    model.load_state_dict(torch.load(BEST_WEIGHTS_PATH, map_location=device, weights_only=True))
    model.eval()
    
    dataset = SyntheticSentenceDataset(DATA_DIR, classes, samples_per_epoch=num_samples, min_words=min_words, max_words=max_words, is_train=False)
    
    raw_wer_tot, filt_wer_tot = 0.0, 0.0
    raw_f1_tot, filt_f1_tot = 0.0, 0.0
    exact_matches = 0
    
    print("\n" + "=" * 90)
    print(f"{'#':<3} | {'GROUND TRUTH':<32} | {'RAW PREDICTION':<26} | {'FILTERED (>=3 frames)':<26}")
    print("=" * 90)
    
    for idx in range(num_samples):
        item = dataset[idx]
        full_tensor, length, labels = item[0], item[1], item[2]
        x = full_tensor.unsqueeze(0).to(device)
        lengths = torch.tensor([length]).to(device)
        
        with torch.no_grad():
            log_probs, pooled_lengths = model(x, lengths)
            log_probs_t = log_probs.transpose(0, 1)
            raw_decoded = ctc_decode(log_probs_t, pooled_lengths, blank=ctc_blank_id)[0]
            filt_decoded = decode_with_min_duration(log_probs, ctc_blank_id, none_idx, min_duration=3)
            
        ref_indices = labels.numpy().tolist()
        ref_words = [classes[i] for i in ref_indices if i != none_idx and i < len(classes)]
        raw_words = [classes[i] for i in raw_decoded if i != none_idx and i < len(classes)]
        filt_words = [classes[i] for i in filt_decoded if i != none_idx and i < len(classes)]
        
        ref_str = " ".join(ref_words) if ref_words else "[SILENCE]"
        raw_str = " ".join(raw_words) if raw_words else "[SILENCE]"
        filt_str = " ".join(filt_words) if filt_words else "[SILENCE]"
        
        raw_wer = compute_wer(ref_words, raw_words)
        filt_wer = compute_wer(ref_words, filt_words)
        raw_f1, _, _ = compute_f1_prec_rec(ref_words, raw_words)
        filt_f1, _, _ = compute_f1_prec_rec(ref_words, filt_words)
        
        raw_wer_tot += raw_wer
        filt_wer_tot += filt_wer
        raw_f1_tot += raw_f1
        filt_f1_tot += filt_f1
        
        if ref_words == filt_words:
            exact_matches += 1
            
        if idx < 15:
            ref_d = (ref_str[:29] + "..") if len(ref_str) > 31 else ref_str
            raw_d = (raw_str[:23] + "..") if len(raw_str) > 25 else raw_str
            filt_d = (filt_str[:23] + "..") if len(filt_str) > 25 else filt_str
            match_sym = "[OK]" if ref_words == filt_words else "    "
            print(f"{idx+1:<3} | {ref_d:<32} | {raw_d:<26} | {filt_d:<26} {match_sym}")
            
    print("=" * 90)
    print(f"Evaluation Results across {num_samples} Synthetic Sentences ({min_words}-{max_words} words):")
    print(f" - Raw CTC WER:             {raw_wer_tot / num_samples:.4f} | Sequence F1: {raw_f1_tot / num_samples:.4f}")
    print(f" - Duration Filtered WER:   {filt_wer_tot / num_samples:.4f} | Sequence F1: {filt_f1_tot / num_samples:.4f}")
    print(f" - Filtered Exact Matches:  {exact_matches}/{num_samples} ({(exact_matches / num_samples)*100.0:.1f}%)")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    evaluate(num_samples=25, max_words=5)
