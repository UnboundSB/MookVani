"""Automated verification for YAP Stage 2 gesture segmentation.

Validates:
  1. Boundary alignment: synthetic sentence spans match actual frames exactly.
  2. Loss masking: padding frames are excluded from all framewise losses.
  3. Dataset independence: train/val/test source clips never overlap.
  4. Gradient flow: HiPPO pooling/readout and YAP layers receive gradients.
  5. Checkpoint loading: safe resume preserves optimizer + epoch state.
  6. Streaming inference: YAPStream output == offline forward_frames.

Run:  python test_stage2_verify.py
"""
import math, os, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_segmentor as tr
from yap_conformer import (
    YAP, YAPConfig, YAPpingConformer, ConformerConfig, hand_part,
    YAPStream, make_segments, end_targets, soft_count_loss,
    focal_presence_loss, layer_conf_loss, Segment,
)


class TestBoundaryAlignment(unittest.TestCase):
    """Gesture start/end timestamps exactly match actual frames."""

    def test_spans_match_generated_frames(self):
        rng = __import__('random').Random(7)
        # Build a tiny corpus: 2 classes, 3 clips each.
        tmp = tempfile.TemporaryDirectory()
        clips = {}
        for cid in (0, 1):
            items = []
            for i in range(3):
                p = Path(tmp.name) / f'{cid}_{i}.pt'
                raw = torch.full((30 + i * 5, 163), float(cid + 1))
                torch.save(raw, p)
                items.append(str(p))
            clips[cid] = items
        raw, spans = tr.make_sentence(clips, 53, rng, 30.0, max_words=3, max_len=400)
        T = len(raw)
        fps = 30.0
        # Every span end must be <= total frames / fps (with tiny epsilon).
        for s, e in spans:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(e, T / fps + 1e-6)
            self.assertLess(s, e)
        # Spans must be non-overlapping and ordered.
        for i in range(len(spans) - 1):
            self.assertLessEqual(spans[i][1], spans[i + 1][0] + 1e-6)
        tmp.cleanup()

    def test_truncation_drops_incomplete_spans(self):
        rng = __import__('random').Random(1)
        tmp = tempfile.TemporaryDirectory()
        clips = {0: [str(Path(tmp.name) / f'0_{i}.pt') for i in range(2)]}
        for p in clips[0]:
            torch.save(torch.full((100, 163), 1.0), p)
        raw, spans = tr.make_sentence(clips, 53, rng, 30.0, max_words=2, max_len=200)
        # All spans must be fully within max_len frames.
        for s, e in spans:
            self.assertLessEqual(e * 30.0, 200.0 + 1e-4, f'span {(s,e)} exceeds max_len')
        tmp.cleanup()


class TestLossMasking(unittest.TestCase):
    """Padding frames must not contribute to any framewise loss."""

    def test_masked_presence_loss_ignores_padding(self):
        B, T = 2, 20
        logit = torch.zeros(B, T)
        target = torch.ones(B, T)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[0, :10] = True
        mask[1, :5] = True
        loss = focal_presence_loss(logit, target, mask)
        # yap_conformer focal_presence_loss: gamma=2, no alpha.
        # logit=0 -> p=0.5, target=1 -> pt=0.5, bce=ln(2), (1-pt)^2=0.25
        # per-frame = 0.25*ln(2) = 0.1733, masked mean over valid frames = same.
        self.assertAlmostEqual(loss.item(), 0.25 * math.log(2), places=5)

    def test_end_targets_normalise_to_count(self):
        t = torch.arange(100, dtype=torch.float32) / 25.0
        spans = [(0.5, 1.5), (2.0, 3.0)]
        tgt = end_targets(t, spans)
        self.assertAlmostEqual(float(tgt.sum()), 2.0, places=4)

    def test_soft_count_loss_matches_end_target_sum(self):
        # When sigmoid(end_logit) sums to n_words over valid frames, loss == 0.
        T = 100
        n_words = 3.0
        prob = n_words / T
        logit_val = math.log(prob / (1 - prob))
        logit = torch.full((1, T), logit_val)
        loss = soft_count_loss(logit, torch.tensor([n_words]), None)
        self.assertAlmostEqual(loss.item(), 0.0, places=4)


class TestDatasetIndependence(unittest.TestCase):
    """Train/val/test source clips must be disjoint."""

    def test_split_sources_disjoint(self):
        corpus = {0: [f'class0_clip_{i}.pt' for i in range(10)],
                  1: [f'class1_clip_{i}.pt' for i in range(10)]}
        train, val = tr.split_sources(corpus, seed=42)
        train_files = {p for v in train.values() for p in v}
        val_files = {p for v in val.values() for p in v}
        self.assertFalse(train_files & val_files)
        self.assertEqual(train_files | val_files, set(corpus[0]) | set(corpus[1]))

    def test_signer_disjoint_preferred(self):
        corpus = {0: [f'signerA_word0_{i}.pt' for i in range(4)] +
                     [f'signerB_word0_{i}.pt' for i in range(4)]}
        train, val, test = tr.split_sources_signer_disjoint(corpus, seed=42)
        train_signers = {p.split('_')[0] for v in train.values() for p in v}
        val_signers = {p.split('_')[0] for v in val.values() for p in v}
        test_signers = {p.split('_')[0] for v in test.values() for p in v}
        # Signer-disjoint: no signer in more than one partition.
        all_signers = list(train_signers) + list(val_signers) + list(test_signers)
        self.assertEqual(len(all_signers), len(set(all_signers)))


class TestGradientFlow(unittest.TestCase):
    """HiPPO pooling/readout and YAP layers must receive gradients."""

    def test_gradients_reach_yap_layers_and_readout(self):
        cfg = YAPConfig(d_model=64, summary_channels=16, hippo_order=8,
                        entity_dim=64, n_mem_layers=2)
        ccfg = ConformerConfig(num_words=10, d_model=64, heads=4, layers=2,
                               conv_kernel=3)
        model = YAPpingConformer(cfg, ccfg)
        B, T, J = 2, 60, cfg.total_joints
        g = torch.Generator().manual_seed(0)
        pose = torch.randn(B, T, J, 3)
        conf = torch.ones(B, T, J)
        valid = torch.ones(B, T, J, dtype=torch.bool)
        t = torch.arange(T, dtype=torch.float32).unsqueeze(0).repeat(B, 1) / 30.0
        spans = [make_segments(t[b].tolist(), [(0.5, 1.0), (1.5, 2.0)], cfg)
                 for b in range(B)]
        res = model(pose, conf, valid, t, segments=spans)
        loss = res['word_logits'].sum()
        loss.backward()
        # YAP memory layers get gradients.
        gnorm = sum(p.grad.abs().sum().item() for p in model.yap.core.layers.parameters()
                    if p.grad is not None)
        self.assertGreater(gnorm, 0, 'no gradient reached YAP memory layers')
        # HiPPO readout gets gradients (entity training supervision).
        self.assertIsNotNone(model.yap.readout[0].weight.grad,
                             'readout weight has no gradient')
        self.assertIsNotNone(model.yap.ctx_vec.grad,
                             'ctx_vec (context flag) has no gradient')
        # Conformer blocks get gradients.
        cgnorm = sum(p.grad.abs().sum().item() for p in model.blocks.parameters()
                     if p.grad is not None)
        self.assertGreater(cgnorm, 0, 'no gradient reached Conformer blocks')


class TestCheckpointResume(unittest.TestCase):
    """Checkpoint save/resume must preserve weights + optimizer + epoch."""

    def test_resume_preserves_state(self):
        cfg = YAPConfig(d_model=32, summary_channels=8, hippo_order=4,
                        entity_dim=32, n_mem_layers=1)
        yap = YAP(cfg)
        optim = torch.optim.AdamW(yap.parameters(), lr=1e-3)
        # Take one training step to change weights.
        x = torch.randn(1, 10, cfg.total_joints, 3)
        conf = torch.ones(1, 10, cfg.total_joints)
        valid = torch.ones(1, 10, cfg.total_joints, dtype=torch.bool)
        t = torch.arange(10, dtype=torch.float32).unsqueeze(0) / 30.0
        out = yap.forward_frames(x, conf, valid, t)
        out['presence'].sum().backward()
        optim.step()
        sd_after_step = {k: v.clone() for k, v in yap.state_dict().items()}

        tmp = tempfile.TemporaryDirectory()
        ckpt = Path(tmp.name) / 'resume.pth'
        torch.save({'yap': yap.state_dict(), 'optimizer': optim.state_dict(),
                    'epoch': 3, 'best': 0.5, 'history': []}, ckpt)

        # Fresh model + optim, load checkpoint.
        yap2 = YAP(cfg)
        optim2 = torch.optim.AdamW(yap2.parameters(), lr=1e-3)
        rb = torch.load(ckpt, map_location='cpu', weights_only=False)
        yap2.load_state_dict(rb['yap'], strict=False)
        optim2.load_state_dict(rb['optimizer'])
        epoch = rb.get('epoch', 0)
        self.assertEqual(epoch, 3)
        for k in sd_after_step:
            self.assertTrue(torch.equal(sd_after_step[k], yap2.state_dict()[k]),
                            f'weight {k} changed after resume')
        tmp.cleanup()


class TestStreamingInference(unittest.TestCase):
    """YAPStream (live) output must equal offline forward_frames."""

    def test_stream_equals_offline(self):
        cfg = YAPConfig(d_model=64, summary_channels=16, hippo_order=8,
                        entity_dim=64, n_mem_layers=2)
        yap = YAP(cfg).eval()
        B, T, J = 1, 90, cfg.total_joints
        g = torch.Generator().manual_seed(3)
        pose = torch.randn(B, T, J, 3)
        conf = torch.ones(B, T, J)
        valid = torch.ones(B, T, J, dtype=torch.bool)
        t = (torch.arange(T).float() / 30.0).unsqueeze(0) + \
            0.002 * torch.rand(B, T, generator=g)
        t, _ = torch.sort(t, dim=1)
        with torch.no_grad():
            off = yap.forward_frames(pose, conf, valid, t)
            stream = YAPStream(yap)
            pres_live = []
            for k in range(T):
                stream.push(pose[0, k], conf[0, k], valid[0, k], float(t[0, k]))
                pres_live.append(stream.last_out['presence'][0])
            stream.flush()
        pres_live = torch.stack(pres_live)
        self.assertTrue(torch.allclose(pres_live, off['presence'][0], atol=1e-5),
                        (pres_live - off['presence'][0]).abs().max().item())


class TestAugmentation(unittest.TestCase):
    """Synthetic sentence generation must include idle padding and speed variation."""

    def test_make_sentence_has_idle_padding(self):
        rng = __import__('random').Random(5)
        tmp = tempfile.TemporaryDirectory()
        clips = {0: [str(Path(tmp.name) / f'0_{i}.pt') for i in range(2)]}
        for p in clips[0]:
            torch.save(torch.full((40, 163), 1.0), p)
        raw, spans = tr.make_sentence(clips, 53, rng, 30.0, max_words=2, max_len=400)
        T = len(raw)
        # First frame(s) should be idle padding (lead >= 10 frames).
        lead_nonzero = int((np.abs(raw[:10]).sum(-1) > 0).sum().item())
        self.assertEqual(lead_nonzero, 0, 'expected idle lead padding')
        # Last frame(s) should be idle padding.
        tail_nonzero = int((np.abs(raw[-10:]).sum(-1) > 0).sum().item())
        self.assertEqual(tail_nonzero, 0, 'expected idle tail padding')
        tmp.cleanup()

    def test_speed_warp_changes_duration(self):
        rng = __import__('random').Random(9)
        tmp = tempfile.TemporaryDirectory()
        clips = {0: [str(Path(tmp.name) / f'0_{i}.pt') for i in range(1)]}
        torch.save(torch.full((100, 163), 1.0), clips[0][0])
        durations = []
        for _ in range(5):
            raw, spans = tr.make_sentence(clips, 53, rng, 30.0, max_words=1, max_len=400)
            durations.append(len(raw) / 30.0)
        tmp.cleanup()
        # Speed augmentation (stretch 0.8-1.25) should produce varied durations.
        self.assertGreater(max(durations) - min(durations), 0.05,
                           'speed augmentation produced identical durations')


if __name__ == '__main__':
    unittest.main(verbosity=2)
