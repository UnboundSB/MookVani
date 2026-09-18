import torch
from itertools import groupby

def ctc_greedy_decode(log_probs_row):
    """log_probs_row: (T, C) log-probs for one sample. Returns list of predicted word ids (blank/dup collapsed)."""
    pred_ids = log_probs_row.argmax(dim=-1).cpu().numpy()
    return [k for k, _ in groupby(pred_ids) if k != 0]

def edit_distance(ref, hyp):
    """Standard Levenshtein distance between two token sequences."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]

def compute_sentence_metrics(model, loader, device):
    """Returns WER (edit-distance based, the standard CSLR metric) AND plain sequence-exact-match
    accuracy (fraction of sentences where the predicted word sequence exactly matches the reference)."""
    model.eval()
    total_errors, total_ref_len = 0, 0
    exact_matches, total_sentences = 0, 0
    examples = []
    
    with torch.no_grad():
        for batch_inputs, batch_targets, in_lens, tgt_lens in loader:
            batch_inputs = batch_inputs.to(device)
            in_lens = in_lens.to(device)
            log_probs, pooled_lens = model(batch_inputs, in_lens)

            for i in range(batch_inputs.size(0)):
                ref = batch_targets[i, :tgt_lens[i]].tolist()
                hyp = ctc_greedy_decode(log_probs[i])
                total_errors += edit_distance(ref, hyp)
                total_ref_len += len(ref)
                total_sentences += 1
                if ref == hyp:
                    exact_matches += 1
                if len(examples) < 5:
                    examples.append((ref, hyp))

    wer = total_errors / max(1, total_ref_len)
    exact_match_acc = exact_matches / max(1, total_sentences)
    return wer, exact_match_acc, examples
