import os
import glob
import random
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.conformer import ISL_Conformer
from data.dataset import ISLSentenceLevelDataset, sentence_collate_fn
from data.utils import resolve_gloss_sequence

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    GLOSS_CSV = "./data/isl_csltr_dataset/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/corpus_csv_files/ISL Corpus sign glosses.csv"
    WORD_FRAMES_DIR = "./data/isl_csltr_dataset/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Word_Level"
    OUTPUT_DIR_SENTENCE = "./data/tensors_sentence_level_163"

    if not os.path.exists(WORD_FRAMES_DIR) or not os.path.exists(GLOSS_CSV):
        print("Dataset not fully extracted or paths are wrong.")
        return

    word_folders = set(os.listdir(WORD_FRAMES_DIR))
    
    sentence_to_words = {}
    df = pd.read_csv(GLOSS_CSV)
    for _, row in df.iterrows():
        sentence = row["Sentence"]
        gloss_str = row["SIGN GLOSSES"]
        if pd.isna(sentence) or pd.isna(gloss_str):
            continue
        resolved, unknown = resolve_gloss_sequence(gloss_str, word_folders)
        if unknown or not resolved:
            continue
        sentence_to_words[sentence] = resolved

    # Need class_to_idx from word training to map words to indices correctly
    # You may need to load this from vocab.json or recreate it
    WORD_DATA_DIR = "./data/tensors_word_level_163"
    all_word_files = glob.glob(f"{WORD_DATA_DIR}/**/*.pt", recursive=True)
    if not all_word_files:
        print("Run word-level extraction and training first.")
        return
    classes = sorted(set(os.path.basename(os.path.dirname(p)) for p in all_word_files))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    word_vocab = {w: i + 1 for w, i in class_to_idx.items()}

    all_sentence_files = glob.glob(f"{OUTPUT_DIR_SENTENCE}/*/*.pt")
    file_phrase_pairs = [(p, os.path.basename(os.path.dirname(p))) for p in all_sentence_files]

    unique_phrases = sorted(set(phrase for _, phrase in file_phrase_pairs))
    random.shuffle(unique_phrases)
    split_point = max(1, int(0.8 * len(unique_phrases)))
    train_phrases = set(unique_phrases[:split_point])
    val_phrases = set(unique_phrases[split_point:])

    train_pairs = [(p, ph) for p, ph in file_phrase_pairs if ph in train_phrases]
    val_pairs = [(p, ph) for p, ph in file_phrase_pairs if ph in val_phrases]

    train_dataset = ISLSentenceLevelDataset(train_pairs, sentence_to_words, word_vocab)
    val_dataset = ISLSentenceLevelDataset(val_pairs, sentence_to_words, word_vocab)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=sentence_collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=sentence_collate_fn, num_workers=2, pin_memory=True)

    model = ISL_Conformer(input_dim=163, num_classes=len(classes)).to(device)
    if os.path.exists("best_isl_conformer_word.pth"):
        model.load_state_dict(torch.load("best_isl_conformer_word.pth", map_location=device))
        print("Loaded Stage 1 weights.")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    EPOCHS = 15
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- [Sentence] Epoch {epoch}/{EPOCHS} ---")
        model.train()
        train_loss, valid_batches = 0.0, 0

        for batch_inputs, batch_targets, in_lens, tgt_lens in tqdm(train_loader, desc="Training"):
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            in_lens = in_lens.to(device)

            optimizer.zero_grad()
            log_probs, pooled_lens = model(batch_inputs, in_lens)
            log_probs = log_probs.permute(1, 0, 2)
            loss = criterion(log_probs, batch_targets, pooled_lens.clamp(min=1), tgt_lens.to(device))

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                train_loss += loss.item()
                valid_batches += 1

        avg_train_loss = train_loss / max(1, valid_batches)

        model.eval()
        val_loss, val_valid_batches = 0.0, 0
        with torch.no_grad():
            for batch_inputs, batch_targets, in_lens, tgt_lens in val_loader:
                batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
                in_lens = in_lens.to(device)
                log_probs, pooled_lens = model(batch_inputs, in_lens)
                log_probs = log_probs.permute(1, 0, 2)
                loss = criterion(log_probs, batch_targets, pooled_lens.clamp(min=1), tgt_lens.to(device))
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    val_valid_batches += 1

        avg_val_loss = val_loss / max(1, val_valid_batches)
        scheduler.step(avg_val_loss)
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_isl_conformer_sentence.pth")
            print("New best sentence-level model saved.")

if __name__ == "__main__":
    train()
