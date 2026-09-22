import os
import sys
import torch
import json
import argparse
import numpy as np

# Add the current directory to sys.path so we can import from local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.isl_sentence_model import ISL_Sentence_Model
from data.extract_video import extract_video_features

def decode_ctc(sequence, idx_to_class, blank_id=0):
    """Decodes a CTC output sequence (removes blanks and repeated tokens)."""
    decoded = []
    last_tok = None
    for tok in sequence:
        if tok != blank_id and tok != last_tok:
            decoded.append(idx_to_class.get(tok, f"<UNK:{tok}>"))
        last_tok = tok
    return " ".join(decoded)

def test_model(input_path, model_path=None, vocab_path="models/word_class_to_idx.json", device="cuda" if torch.cuda.is_available() else "cpu"):
    if not os.path.exists(vocab_path):
        print(f"Error: Vocab file {vocab_path} not found.")
        return

    with open(vocab_path, "r") as f:
        class_to_idx = json.load(f)
    
    # +1 because CTC blank is 0, so words are 1-indexed
    idx_to_class = {i + 1: c for c, i in class_to_idx.items()}
    num_classes = len(class_to_idx)
    print(f"Loaded vocabulary with {num_classes} classes.")

    # Auto-detect model
    if model_path is None:
        if os.path.exists("models/best_sentence_model.pth"):
            model_path = "models/best_sentence_model.pth"
        elif os.path.exists("models/best_synthetic_model.pth"):
            model_path = "models/best_synthetic_model.pth"
        else:
            print("Error: No pre-trained sentence model found in models/")
            return
            
    print(f"Loading model weights from: {model_path}")
    
    # freeze_base=False is important here so all gradients are disabled normally in eval
    model = ISL_Sentence_Model(num_classes=num_classes, d_model=256, lstm_layers=1, dropout=0.0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Process Input
    if input_path.endswith(".pt"):
        print(f"Loading pre-extracted tensor: {input_path}")
        tensor = torch.load(input_path, weights_only=True).to(device)
    elif input_path.endswith(".mp4") or input_path.endswith(".avi") or input_path.endswith(".mov") or input_path.endswith(".mkv"):
        print(f"Extracting MediaPipe features from video: {input_path}")
        tensor = extract_video_features(input_path)
        if tensor is None:
            print("Failed to extract features from video. Is MediaPipe able to see hands?")
            return
        tensor = tensor.to(device)
    else:
        print("Error: Input must be a .pt tensor file or a video file (.mp4/.avi/.mov/.mkv).")
        return

    # Add batch dimension
    tensor = tensor.unsqueeze(0) # (1, T, 163)
    lengths = torch.tensor([tensor.size(1)], dtype=torch.long).to(device)

    # Inference
    print(f"Running inference on sequence of length {tensor.size(1)}...")
    with torch.no_grad():
        log_probs, pooled_lens = model(tensor, lengths)
        
    # log_probs is (1, T_pooled, num_classes+1)
    probs = torch.exp(log_probs)
    predicted_seq = probs.argmax(dim=-1).squeeze(0).cpu().numpy() # (T_pooled,)

    print(f"\nRaw model output (Argmax Sequence):\n{predicted_seq}")
    
    final_sentence = decode_ctc(predicted_seq, idx_to_class, blank_id=0)
    print(f"\n======================================")
    print(f"🗣️ PREDICTED SENTENCE: {final_sentence}")
    print(f"======================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the ISL Sentence Model")
    parser.add_argument("input_path", type=str, help="Path to video file (.mp4) or tensor file (.pt)")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights (.pth)")
    args = parser.parse_args()
    
    test_model(args.input_path, model_path=args.model)
