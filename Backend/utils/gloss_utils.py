import os
import pandas as pd
from collections import Counter

GLOSS_ALIASES = {
    # merged / compound folders (a folder covers 2+ gloss tokens signed together)
    "HI": "HELLO_HI",
    "I": "I_ME_MINE_MY", "ME": "I_ME_MINE_MY", "MY": "I_ME_MINE_MY",
    "LOVE": "LIKE_LOVE",
    "COLLEGE": "COLLEGE_SCHOOL", "SCHOOL": "COLLEGE_SCHOOL",
    "DONOT": "NOT", "DONT": "NOT",
    "OLD": "OLD_AGE",
    "SOMEONE": "SOME ONE",
    # spelling/typo/tense variants
    "HIDE": "HIDING",
    "CRY": "CRYING",
    "STOPPED": "STOP",
    "ENJOYED": "ENJOY",
    "COMING": "COME",
    "CONGRATULATIIONS": "CONGRATULATIONS",
    # punctuation/artifact cleanup -> map to the clean token, re-resolved recursively
    "ANYTHING,": "ANYTHING",
    "FINE.": "FINE",
    "MEDICINE,": "MEDICINE",
}

GLOSS_BIGRAM_ALIASES = {
    ("TAKE", "CARE"): "TAKE CARE",
    ("TAKE", "TIME"): "TAKE TIME",
    ("DON'T", "CARE"): "DON'T CARE",
    ("DONT", "CARE"): "DON'T CARE",
}

GLOSS_DROP_TOKENS = {
    "(AGE)", "XXXXXXXX",
    "A", "ABOUT", "AM", "ANY", "BE", "BY", "CAME", "CAREER", "GLASS", "GOT", "HAIR",
    "HAVE", "HE", "HIM", "IN", "INTO", "IT", "KNOW", "LET", "LIGHT", "LOT", "MAKE",
    "MEAN", "MORE", "MUCH", "NEED", "NEVER", "NO", "NOW", "OF", "OFF", "ON", "ONE",
    "ONWARDS", "PLAN", "SHE", "SIR", "SO", "SOME", "SOMEHOW", "STOPPED_DUPLICATE",
    "SUFFERING", "THE", "THERE", "THIS", "TIME", "TO", "TRY", "TURN", "VERY", "WAY",
    "WE", "WHEN", "WHICH", "WHY", "WITH", "YOUR", "YOURSELF",
}

def resolve_gloss_sequence(gloss_string, word_folders, log_drops=None):
    """Convert a raw 'SIGN GLOSSES' string into a list of real Frames_Word_Level folder names.
    Applies bigram aliases first, then unigram aliases, then drops confirmed function words/artifacts.
    Any token that's still unresolved (genuinely unknown) causes the WHOLE sentence to be skipped
    (logged), since an unrecognized token is a sign we can't confidently confirm is safe to drop.
    """
    tokens = gloss_string.strip().split()
    resolved = []
    i = 0
    unknown = []
    while i < len(tokens):
        # Try bigram merge first
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) in GLOSS_BIGRAM_ALIASES:
            folder = GLOSS_BIGRAM_ALIASES[(tokens[i], tokens[i + 1])]
            resolved.append(folder)
            i += 2
            continue

        tok = tokens[i]
        if tok in word_folders:
            resolved.append(tok)
        elif tok in GLOSS_ALIASES:
            aliased = GLOSS_ALIASES[tok]
            if aliased in word_folders:
                resolved.append(aliased)
            else:
                unknown.append(tok)  # alias target itself doesn't exist — flag it
        elif tok in GLOSS_DROP_TOKENS:
            if log_drops is not None:
                log_drops.append(tok)
        else:
            unknown.append(tok)
        i += 1

    return resolved, unknown

def load_sentence_to_words(gloss_csv_path, word_frames_dir):
    word_folders = set(os.listdir(word_frames_dir)) if os.path.exists(word_frames_dir) else set()
    sentence_to_words = {}
    drop_log = []
    skipped_sentences = []

    if os.path.exists(gloss_csv_path):
        df = pd.read_csv(gloss_csv_path)
        for _, row in df.iterrows():
            sentence = row["Sentence"]
            gloss_str = row["SIGN GLOSSES"]
            if pd.isna(sentence) or pd.isna(gloss_str):
                continue

            this_drops = []
            resolved, unknown = resolve_gloss_sequence(gloss_str, word_folders, log_drops=this_drops)

            if unknown:
                skipped_sentences.append((sentence, gloss_str, unknown))
                continue

            if not resolved:
                skipped_sentences.append((sentence, gloss_str, ["<all tokens dropped>"]))
                continue

            sentence_to_words[sentence] = resolved
            drop_log.extend(this_drops)

        print(f"✅ Resolved gloss sequences for {len(sentence_to_words)}/{len(df)} sentences.")
        if skipped_sentences:
            print(f"⚠️ Skipped {len(skipped_sentences)} sentences due to unresolved tokens.")
            for sentence, gloss_str, unknown in skipped_sentences[:5]:
                print(f"   '{sentence}' (gloss: '{gloss_str}') -> unknown: {unknown}")
    else:
        print(f"❌ {gloss_csv_path} not found.")

    return sentence_to_words, skipped_sentences, drop_log
