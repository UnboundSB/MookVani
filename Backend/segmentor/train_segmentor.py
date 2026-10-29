"""YAP-only gesture boundary pretraining. No Conformer, CTC or word classifier.

Use existing segmentor/data/<class>/**/*.pt word recordings (T,163).
Labels only identify source recordings; the objective is presence/end segmentation.
"""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from yap_conformer import YAP, YAPConfig, presence_targets, end_targets, focal_presence_loss, layer_conf_loss

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent


def discover(root: Path, classes: list[str], max_per_class: int, seed: int):
    rng = random.Random(seed)
    result = {}
    for cid, name in enumerate(classes):
        folder = root / name
        if not folder.is_dir():
            continue
        paths = [p for p in folder.rglob('*.pt') if p.name != 'processed.pt']
        rng.shuffle(paths)
        good = []
        for path in paths:
            try:
                obj = torch.load(path, map_location='cpu', weights_only=True)
                if isinstance(obj, torch.Tensor) and obj.numel() >= 163 * 4 and obj.numel() % 163 == 0:
                    good.append(path)
            except (RuntimeError, ValueError, OSError, TypeError):
                continue
            if len(good) == max_per_class:
                break
        if good:
            result[cid] = good
    return result


def split_sources(corpus, seed, train_fraction=0.8):
    rng = random.Random(seed)
    train, val = {}, {}
    for key, sources in corpus.items():
        items = sources.copy()
        rng.shuffle(items)
        if len(items) < 2:
            train[key] = items
            continue
        n = min(len(items)-1, max(1, int(len(items) * train_fraction)))
        train[key], val[key] = items[:n], items[n:]
    return train, val


def read_xyz(path: Path, joints: int):
    clip = torch.load(path, map_location='cpu', weights_only=True).float()
    clip = clip.reshape(-1, 163)
    if joints * 3 > 163:
        raise ValueError('Only the 53-joint corpus layout is supported by these tensors')
    return clip[:, :joints * 3].reshape(-1, joints, 3).numpy().copy()


def stretch(clip, rng):
    length = max(4, round(len(clip) * rng.uniform(0.8, 1.25)))
    pos = np.linspace(0, len(clip)-1, length)
    low = pos.astype(np.int64)
    high = np.minimum(low + 1, len(clip)-1)
    a = (pos - low).astype(np.float32)[:, None, None]
    return (clip[low]*(1-a) + clip[high]*a).astype(np.float32)


def make_sentence(corpus, joints, rng, fps, max_words, max_len):
    """Use actual frame offsets, non-overlapping clips, and real inter-word pauses.

    No synthetic crossfades: naive blend-label supervision is ambiguous. Real
    continuous clips are needed to learn true coarticulated gesture transitions.
    """
    keys = list(corpus)
    n_words = rng.randint(1, max_words)
    frames = []
    spans = []
    cursor = 0
    lead = rng.randint(10, 25)
    frames.append(np.zeros((lead, joints, 3), np.float32))
    cursor += lead
    for k in range(n_words):
        path = rng.choice(corpus[rng.choice(keys)])
        clip = stretch(read_xyz(path, joints), rng)
        # No artificial idle frames between some word pairs.
        if k:
            gap = rng.choice((0, 0, rng.randint(2, 7), rng.randint(10, 25)))
            if gap:
                frames.append(np.zeros((gap, joints, 3), np.float32))
                cursor += gap
        start = cursor
        frames.append(clip)
        cursor += len(clip)
        spans.append((start, cursor))  # half-open frame indexes
    tail = rng.randint(10, 25)
    frames.append(np.zeros((tail, joints, 3), np.float32))
    raw = np.concatenate(frames, axis=0)
    if len(raw) > max_len:
        raw = raw[:max_len]
        # Only train against fully observed spans. No contradictory partial counts.
        spans = [(s, e) for s, e in spans if e <= max_len]
    if not spans:
        raise ValueError('max_len too short for first complete gesture')
    return raw, [(s/fps, e/fps) for s,e in spans]


class SentenceSet(Dataset):
    def __init__(self, clips, joints, count, fps, max_words, max_len, seed, training):
        self.clips = clips
        self.joints = joints
        self.count = count
        self.fps = fps
        self.max_words = max_words
        self.max_len = max_len
        self.seed = seed
        self.training = training
        self.epoch = 0

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        rng = random.Random(self.seed + index * 1000003 + (self.epoch * 1000000007 if self.training else 0))
        for _ in range(20):
            try:
                raw, spans = make_sentence(self.clips, self.joints, rng, self.fps, self.max_words, self.max_len)
                break
            except ValueError:
                continue
        else:
            raise ValueError('Could not generate a fully visible word; increase --max-len')
        pose = torch.from_numpy(raw)
        T = len(pose)
        valid = torch.isfinite(pose).all(-1) & (pose.abs().sum(-1) > 0)
        pose = torch.nan_to_num(pose, nan=0.0, posinf=0.0, neginf=0.0)
        conf = valid.float()
        if self.training:
            pose += torch.randn_like(pose) * 0.002 * valid.unsqueeze(-1)
            if rng.random() < 0.25 and T >= 8:
                s = rng.randrange(T-4)
                e = min(T, s+rng.randint(2,6))
                part = slice(0,21) if rng.random() < 0.5 else slice(21,42)
                valid[s:e, part] = False
                conf[s:e, part] = 0
                pose[s:e, part] = 0
        times = torch.arange(T, dtype=torch.float32) / self.fps
        return pose, conf, valid, times, spans


def collate(items):
    B = len(items)
    T = max(len(item[0]) for item in items)
    J = items[0][0].shape[1]
    pose = torch.zeros(B,T,J,3)
    conf = torch.zeros(B,T,J)
    valid = torch.zeros(B,T,J,dtype=torch.bool)
    t = torch.zeros(B,T)
    mask = torch.zeros(B,T,dtype=torch.bool)
    spans = []
    for b,(p,c,v,tt,ss) in enumerate(items):
        n = len(p)
        pose[b,:n],conf[b,:n],valid[b,:n],t[b,:n] = p,c,v,tt
        t[b,n:] = tt[-1]
        mask[b,:n] = True
        spans.append(ss)
    return pose,conf,valid,t,mask,spans


def state_from_checkpoint(path):
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(obj,dict):
        raise ValueError('Checkpoint is not a state dictionary')
    for key in ('yap', 'model', 'state_dict'):
        if key in obj and isinstance(obj[key],dict):
            obj = obj[key]
    if any(k.startswith('module.') for k in obj):
        obj = {k.removeprefix('module.'):v for k,v in obj.items()}
    if any(k.startswith('yap.') for k in obj):
        obj = {k.removeprefix('yap.'):v for k,v in obj.items() if k.startswith('yap.')}
    return obj


def losses(yap, batch, device, return_output=False):
    pose, conf, valid, t, mask, spans = batch
    pose,conf,valid,t,mask = (x.to(device) for x in (pose,conf,valid,t,mask))
    output = yap.forward_frames(pose,conf,valid,t)
    B,T = mask.shape
    presence = torch.zeros(B,T,device=device)
    ending = torch.zeros(B,T,device=device)
    for b,ss in enumerate(spans):
        n = int(mask[b].sum().item())
        presence[b,:n] = presence_targets(t[b,:n], ss)
        ending[b,:n] = end_targets(t[b,:n], ss)
    pmask = mask.float()
    pres = focal_presence_loss(output['presence'],presence,pmask)
    end_raw = F.binary_cross_entropy_with_logits(output['end'],ending,reduction='none',
                                                 pos_weight=pose.new_tensor(10.0))
    end = (end_raw*pmask).sum()/pmask.sum().clamp_min(1)
    layer = layer_conf_loss(output['layer_conf'],presence,pmask)
    total = pres + end + 0.5*layer
    parts = {'presence':pres.detach().item(), 'end':end.detach().item(),
             'layer':layer.detach().item()}
    return (total, parts, output) if return_output else (total, parts)



def boundary_matches(pred_times, true_times, tolerance_s):
    """One-to-one time matching; duplicate detections count as false positives."""
    pairs = sorted((abs(p - g), i, j)
                   for i, p in enumerate(pred_times)
                   for j, g in enumerate(true_times)
                   if abs(p - g) <= tolerance_s)
    used_p, used_g = set(), set()
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    return len(used_p)


def boundary_counts(yap, output, times, mask, spans, tolerance_s=0.2):
    """Evaluate the REAL production SegmentTracker, not thresholded end logits.

    Report starts and ends separately, and a combined boundary F1.
    `eos` segments are excluded as predictions: a forced end of recording is
    not evidence of detecting a gesture end.
    """
    lengths = mask.sum(dim=1).long()
    predicted = yap.segment_batch(output, times, lengths)
    result = dict(matches=0, predicted=0, truth=0,
                  start_matches=0, start_predicted=0, start_truth=0,
                  end_matches=0, end_predicted=0, end_truth=0,
                  reasons={})
    for segments, gt in zip(predicted, spans):
        # EOS is artificial. Do not count it as a learned cut.
        segments = [seg for seg in segments if seg.reason != 'eos']
        pred_starts = [float(seg.start_t) for seg in segments]
        pred_ends = [float(seg.end_t) for seg in segments]
        gt_starts = [float(start) for start, _ in gt]
        gt_ends = [float(end) for _, end in gt]
        for kind, ps, gs in (('start', pred_starts, gt_starts),
                             ('end', pred_ends, gt_ends)):
            matches = boundary_matches(ps, gs, tolerance_s)
            result[f'{kind}_matches'] += matches
            result[f'{kind}_predicted'] += len(ps)
            result[f'{kind}_truth'] += len(gs)
            result['matches'] += matches
            result['predicted'] += len(ps)
            result['truth'] += len(gs)
        for seg in segments:
            result['reasons'][seg.reason] = result['reasons'].get(seg.reason, 0) + 1
    return result


def prf(count, prefix=''):
    m = count[f'{prefix}matches']
    p = count[f'{prefix}predicted']
    g = count[f'{prefix}truth']
    precision = m / p if p else 0.0
    recall = m / g if g else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'matched': m, 'predicted': p, 'ground_truth': g}


def run(args):
    if args.max_len < 16 or args.fps <= 0 or args.max_words < 1:
        raise ValueError('Invalid dataset length, fps, or max_words')
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with open(args.classes, encoding='utf-8') as f:
        classes = json.load(f)
    cfg = YAPConfig.corpus(d_model=args.d_model, n_mem_layers=args.n_mem_layers,
                           summary_channels=args.summary_channels,hippo_order=args.hippo_order,
                           entity_dim=args.entity_dim)
    yap = YAP(cfg).to(device)
    if args.init:
        pretrained = state_from_checkpoint(Path(args.init))
        own = yap.state_dict()
        compatible = {k:v for k,v in pretrained.items() if k in own and own[k].shape == v.shape}
        if not compatible:
            raise RuntimeError('ZERO compatible YAP tensors in initialization checkpoint')
        missing,unexpected = yap.load_state_dict(compatible,strict=False)
        print(f'[INIT] loaded {len(compatible)}/{len(own)} tensors; missing={len(missing)} unexpected={len(unexpected)}')
    clips = discover(Path(args.dataset_root),classes,args.max_per_class,args.seed)
    train_sources,val_sources = split_sources(clips,args.seed)
    if not train_sources or not val_sources:
        raise RuntimeError('Train or validation source recordings missing')
    out = Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'yap_phaseA_best.pth').exists() and not args.overwrite:
        raise FileExistsError('Output checkpoint exists. Use a new --out-dir or explicitly --overwrite')
    (out/'source_split.json').write_text(json.dumps({
        'train':{str(k):[str(p) for p in v] for k,v in train_sources.items()},
        'val':{str(k):[str(p) for p in v] for k,v in val_sources.items()}},indent=2),encoding='utf-8')
    train = SentenceSet(train_sources,cfg.total_joints,args.train_sentences,args.fps,args.max_words,args.max_len,args.seed,True)
    val = SentenceSet(val_sources,cfg.total_joints,args.val_sentences,args.fps,args.max_words,args.max_len,args.seed+7,False)
    tr = DataLoader(train,batch_size=args.batch_size,shuffle=True,collate_fn=collate,num_workers=0)
    va = DataLoader(val,batch_size=args.batch_size,shuffle=False,collate_fn=collate,num_workers=0)
    params = [p for p in yap.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,lr=args.lr,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    best = math.inf
    history=[]
    # These names, loss definitions and checkpoint schema remain unchanged.
    loss_names = ('presence', 'end', 'layer')
    descriptions = {
        'presence': 'focal presence loss (gesture-active frames)',
        'end': 'masked weighted BCE (gesture end boundaries)',
        'layer': 'layer confidence loss (deep supervision; weight 0.5)',
    }
    for epoch in range(args.epochs):
        train.epoch=epoch
        yap.train()
        train_sum=0.0
        train_parts={k:0.0 for k in loss_names}
        for batch in tqdm(tr,desc=f'YAP {epoch+1}/{args.epochs}',leave=False):
            loss,parts = losses(yap,batch,device)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite YAP loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1.0)
            optimizer.step()
            train_sum+=loss.item()
            for k in loss_names:
                train_parts[k]+=parts[k]
        yap.eval()
        val_sum=0.0
        metrics={k:0.0 for k in loss_names}
        boundary_totals = dict(matches=0, predicted=0, truth=0,
                               start_matches=0, start_predicted=0, start_truth=0,
                               end_matches=0, end_predicted=0, end_truth=0,
                               reasons={})
        with torch.no_grad():
            for batch in va:
                loss, parts, output = losses(yap, batch, device, return_output=True)
                val_sum += loss.item()
                for k in metrics:
                    metrics[k] += parts[k]
                # Same validation inputs, same forward pass. Uses production tracker.
                _, _, _, times, mask, spans = batch
                bc = boundary_counts(yap, output, times, mask, spans)
                for k in boundary_totals:
                    if k == 'reasons':
                        for reason, n in bc[k].items():
                            boundary_totals[k][reason] = boundary_totals[k].get(reason, 0) + n
                    else:
                        boundary_totals[k] += bc[k]
        boundary_all = prf(boundary_totals)
        boundary_start = prf(boundary_totals, 'start_')
        boundary_end = prf(boundary_totals, 'end_')
        tr_loss=train_sum/len(tr)
        v_loss=val_sum/len(va)
        train_avg={k:train_parts[k]/len(tr) for k in loss_names}
        val_avg={k:metrics[k]/len(va) for k in loss_names}
        history.append({'epoch':epoch+1,'train':tr_loss,'val':v_loss,
                        **{f'val_{k}':val_avg[k] for k in loss_names},
                        **{f'train_{k}':train_avg[k] for k in loss_names},
                        'boundary_f1_200ms':boundary_all['f1'],
                        'boundary_precision_200ms':boundary_all['precision'],
                        'boundary_recall_200ms':boundary_all['recall'],
                        'boundary_start_f1_200ms':boundary_start['f1'],
                        'boundary_end_f1_200ms':boundary_end['f1'],
                        'boundary_cut_reasons':boundary_totals['reasons']})
        print('\n'+'='*76)
        print(f'  YAP SEGMENTATION ONLY | EPOCH {epoch+1:03d}/{args.epochs:03d} | LR {optimizer.param_groups[0]["lr"]:.2e}')
        print('-'*76)
        print(f'  {"Loss (definition)":<52} {"Train":>9} {"Validation":>11}')
        print('-'*76)
        for k in loss_names:
            print(f'  {descriptions[k]:<52} {train_avg[k]:>9.4f} {val_avg[k]:>11.4f}')
        print('-'*76)
        print(f'  {"TOTAL (presence + end + 0.5 * layer)":<52} {tr_loss:>9.4f} {v_loss:>11.4f}')
        improved=v_loss<best
        if improved:
            best=v_loss
            torch.save({'yap':yap.state_dict(),'epoch':epoch+1,'history':history},out/'yap_phaseA_best.pth')
        print('-'*76)
        print('  BOUNDARY METRICS (validation, production SegmentTracker, +/-200 ms)')
        print(f'  F1 (start + end boundaries): {boundary_all["f1"]:.4f} | '
              f'Precision: {boundary_all["precision"]:.4f} | Recall: {boundary_all["recall"]:.4f}')
        print(f'  Start F1: {boundary_start["f1"]:.4f} | '
              f'End F1: {boundary_end["f1"]:.4f}')
        print(f'  Matched: {boundary_all["matched"]} / Predicted: {boundary_all["predicted"]}'
              f' / Ground truth: {boundary_all["ground_truth"]}')
        print(f'  Cut reasons: {boundary_totals["reasons"]}')
        print(f'  Best validation: {best:.4f}'+(' [new checkpoint saved]' if improved else ''))
        print('='*76+'\n',flush=True)
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        scheduler.step()
    torch.save(yap.state_dict(),out/'yap_phaseA_final.pth')
    print(f'[DONE] best validation loss={best:.4f}; no Conformer or CTC used')


def arguments():
    p=argparse.ArgumentParser(description='YAP-only segmentation pretraining (no word classifier)')
    p.add_argument('--dataset-root',type=Path,default=HERE/'data')
    p.add_argument('--classes',type=Path,default=BACKEND/'models'/'word_vocabulary'/'classes.json')
    p.add_argument('--init',type=Path,default=HERE/'models'/'yap_stage1'/'yap_noise_robust.pth')
    p.add_argument('--out-dir',type=Path,default=HERE/'models'/'yap_segmentation_only_v1')
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--train-sentences',type=int,default=4000)
    p.add_argument('--val-sentences',type=int,default=400)
    p.add_argument('--max-per-class',type=int,default=22)
    p.add_argument('--max-words',type=int,default=5)
    p.add_argument('--max-len',type=int,default=900)
    p.add_argument('--fps',type=float,default=30.)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--d-model',type=int,default=128)
    p.add_argument('--n-mem-layers',type=int,default=3)
    p.add_argument('--summary-channels',type=int,default=32)
    p.add_argument('--hippo-order',type=int,default=16)
    p.add_argument('--entity-dim',type=int,default=256)
    p.add_argument('--overwrite',action='store_true')
    p.add_argument('--cpu',action='store_true')
    return p.parse_args()


if __name__=='__main__':
    run(arguments())
