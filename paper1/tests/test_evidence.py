import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import torch
from torch import nn

from legolm_p01.evaluate import bpb, variation, accuracy_summary,ranking
from legolm_p01.hosts import Host, LoRA
from legolm_p01.legacy.runtime_lm import QwenModel,QwenConfig
from legolm_p01.provenance import file_hash
from legolm_p01.train import Recipe,run


class EvidenceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_bpb_sums_bytes_and_excludes_special_targets(self):
        host=Mock()
        host.device=torch.device("cpu")
        host.tokens.widths=torch.tensor([0,1,2,3])
        host.losses.return_value=torch.tensor([[2.,6.],[999.,4.]])
        result=bpb(host,None,"none",torch.tensor([[1,2,3],[1,0,2]]),torch.zeros(3,4),torch.zeros(4))
        self.assertEqual(result["bytes"],7)
        self.assertAlmostEqual(result["bpb"],12/(math.log(2)*7))

    def test_effective_geometry_matches_explicit_dense_updates(self):
        torch.manual_seed(3)
        vectors=torch.randn(5,14)
        class Fixed(nn.Module):
            def forward(self,x):
                return vectors
        host=Mock()
        host.device="cpu"
        host.target_manifest.return_value=[{"module":"test","a_shape":[3,2],"b_shape":[2,4],"offset":0}]
        result=variation(host,Fixed(),torch.zeros(5,1))
        dense=(vectors[:,:6].view(5,3,2)@vectors[:,6:].view(5,2,4)).flatten(1)
        norms=nn.functional.normalize(dense,dim=1)
        mask=~torch.eye(5,dtype=torch.bool)
        self.assertAlmostEqual(result["effective_cosine"],float((norms@norms.T)[mask].mean()),places=5)
        self.assertAlmostEqual(result["effective_centered_energy"],float((dense-dense.mean(0)).square().sum(1).mean()),places=4)

    def test_class_balance_is_separate_from_micro_accuracy(self):
        scores=[[1,0],[1,0],[1,0],[1,0]]
        result=accuracy_summary(scores,[0,0,0,1])
        self.assertEqual(result["micro_accuracy"],.75)
        self.assertEqual(result["macro_accuracy"],.5)

    def test_input_independent_ranking_retains_the_full_population(self):
        host=Mock();host.device=torch.device("cpu")
        host.tokens.task_encode.side_effect=lambda s:{"P ":[1,2],"P a":[1,3],"P b":[1,4]}[s]
        host.tokens.widths=torch.ones(5,dtype=torch.long)
        host.losses.return_value=torch.tensor([[.1],[.8]])
        dataset={"prompt":"P ","categories":["a","b"],"test":{"features":torch.zeros(2000,3),
            "labels":[0,1]*1000,"groups":[str(i//2) for i in range(2000)],"ids":list(map(str,range(2000)))}}
        result=ranking(host,None,"none",dataset,torch.zeros(3),sensor_batch=4)
        self.assertEqual(len(result["items"]),2000)
        self.assertEqual(result["summary"]["micro_accuracy"],.5)
        self.assertEqual(result["summary"]["macro_accuracy"],.5)
        host.losses.assert_called_once()

    def test_real_optimizer_resume_matches_uninterrupted_weights(self):
        def tiny(*args,**kwargs):
            cfg=QwenConfig(sequence_len=256,vocab_size=32,n_layer=1,n_head=2,n_kv_head=1,n_embd=16,intermediate_size=32,head_dim=8)
            model=QwenModel(cfg).eval().requires_grad_(False)
            base=model.transformer.h[0].mlp.up_proj
            target=LoRA(base)
            model.transformer.h[0].mlp.up_proj=target
            tokens=Mock();tokens.widths=torch.ones(32,dtype=torch.long)
            return Host("mini",model,16,tokens,{"fixture":"tiny-frozen-host"},[("transformer.h.0.mlp.up_proj",target)])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);features=root/"features";text=root/"text"
            features.mkdir();text.mkdir()
            torch.manual_seed(5)
            dataset={"modality":"audio","categories":["a","b"],"prompt":"P ",
                     "train":{"features":torch.randn(16,4),"ids":list(map(str,range(16))),"labels":[0,1]*8},
                     "test":{"features":torch.randn(4,4),"ids":list(map(str,range(4))),"labels":[0,1]*2},
                     "unpaired":{"features":torch.randn(16,4),"ids":list(map(str,range(16)))}}
            torch.save(dataset,features/"audio.pt")
            torch.save({"train":torch.randint(0,32,(32,129)),"val":torch.randint(0,32,(8,129))},text/"fixture.pt")
            (text/"text-mini.json").write_text(json.dumps({"file":"fixture.pt","sha256":file_hash(text/"fixture.pt")}))
            recipe=Recipe(host="mini",cohort="U",modality="audio",steps=4)
            with patch("legolm_p01.train.load_host",side_effect=tiny):
                full=run(recipe,root,root,features,text,root/"full","cpu")
                run(recipe,root,root,features,text,root/"resume","cpu",stop_after=2)
                resumed=run(recipe,root,root,features,text,root/"resume","cpu")
            def weights(result):
                path=Path(result["directory"])/result["checkpoints"][-1]["path"]
                return torch.load(path,weights_only=False)["adapter"]
            for key,value in weights(full).items():
                self.assertTrue(torch.equal(value,weights(resumed)[key]),key)
            with patch("legolm_p01.train.load_host",side_effect=tiny),patch("legolm_p01.train.backend_identity",return_value={"different_backend":True}):
                changed=run(recipe,root,root,features,text,root/"full","cpu")
            self.assertNotEqual(full["training_id"],changed["training_id"])


if __name__=="__main__":
    unittest.main()
