"""Unit tests for corrected synthetic data, splits, masks, and device safety.
Run: python -m unittest test_stage2_fixes.py
"""
import sys, types, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
# When running standalone outside the original repository, provide stubs for
# helper modules that are only needed by full model training.
try:
    import utils.checkpoint_utils
except ImportError:
    sys.modules['utils']=types.ModuleType('utils')
    m=types.ModuleType('utils.checkpoint_utils')
    m.load_flexible=lambda *a, **kw: ([],[])
    sys.modules['utils.checkpoint_utils']=m
try:
    import noise_aug
except ImportError:
    m=types.ModuleType('noise_aug');m.add_noise=lambda *a, **kw: None
    sys.modules['noise_aug']=m
import train_segmentor as tr

class Stage2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.clips={}
        for cid in (0,1):
            items=[]
            for i in range(4):
                path=self.root/f'{cid}_{i}.pt'
                raw=torch.full((25+i,163),float(cid+1))
                torch.save(raw,path)
                items.append((str(path),str(path),cid))
            self.clips[cid]=items
    def tearDown(self): self.tmp.cleanup()
    def test_split_independent_and_reproducible(self):
        a,b=tr.split_sources(self.clips,42,train_fraction=0.75)
        a2,b2=tr.split_sources(self.clips,42,train_fraction=0.75)
        self.assertEqual(a,a2);self.assertEqual(b,b2)
        self.assertFalse(set(str(c[0]) for v in a.values() for c in v)&set(str(c[0]) for v in b.values() for c in v))
    def test_spans_align_with_final_tensor(self):
        rng=__import__('random').Random(0)
        keys=list(self.clips)
        corpus={k:[p for p,_ in v] for k,v in self.clips.items()}
        raw,spans_t=tr.make_sentence(corpus,53,rng,30.0,max_words=4,max_len=45)
        self.assertEqual(raw.shape[1],159)
        self.assertEqual(len(spans_t),4)
        self.assertTrue(all(0 <= s < e <= len(raw)/30+1e-5 for s,e in spans_t))
        self.assertTrue(all(spans_t[i][0] <= spans_t[i+1][0] for i in range(3)))
    def test_validation_deterministic(self):
        d=tr.SentenceSet(self.clips,53,10,30.0,4,45,42,False)
        x=d[4]; y=d[4]
        self.assertTrue(torch.equal(x[0],y[0]));self.assertEqual(x[4:],y[4:])
    def test_truncation_discards_incomplete_labels(self):
        d=tr.SentenceSet(self.clips,53,10,30.0,5,45,42,False)
        for i in range(10):
            pose,_,_,_,spans=d[i]
            self.assertTrue(all(e <= len(pose)/30+1e-4 for _,e in spans))
    def test_npy_clip(self):
        path=self.root/'x.npy';np.save(path,np.ones((15,163),dtype=np.float32))
        # _load_clip does not exist in train_segmentor; use split_163_to_pose directly
        from yap_conformer import split_163_to_pose
        x163=torch.load(path,weights_only=True).float()
        pose,_,_=split_163_to_pose(x163,53)
        self.assertEqual(pose.shape,(15,53,3))

if __name__=='__main__': unittest.main()
