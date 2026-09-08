import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from legolm_p01.checkpoints import evidence_checkpoints, validate_index, materialize, fetch
from legolm_p01.provenance import digest, file_hash, write_json


class CheckpointDistribution(unittest.TestCase):
    def fixture(self, root):
        recipe={"host":"qwen08","cohort":"U","modality":"audio","kind":"static",
                "feature_mode":"true","seed":42,"diversity":0.}
        write_json(root/"training.json",{"recipe":recipe,"training_id":"training",
                   "checkpoints":[{"sha256":"a"*64,"checkpoint_bytes":12,"step":300},
                                  {"sha256":"b"*64,"checkpoint_bytes":16,"step":3000}]})
        write_json(root/"index.json",{"trajectories":[{"path":"training.json"}]})
        metadata={}
        caches=[]
        for modality, char in zip(("imu","audio","vision"),("c","d","e")):
            path=root/"data"/(modality+".json")
            write_json(path,{"cache_sha256":char*64})
            metadata["data/"+path.name]={"sha256":file_hash(path),"bytes":path.stat().st_size}
            caches.append({"modality":modality,"sha256":char*64,"bytes":10,
                           "url":"https://example.test/weights/"+char*64+".pt"})
        write_json(root/"manifest.json",{"matrix_id":"matrix","status":"complete","files":metadata})
        index={"schema":"paper1-checkpoint-index-v2","matrix_id":"matrix",
               "evidence_manifest_sha256":file_hash(root/"manifest.json"),
               "checkpoints":[{**entry,"url":"https://example.test/weights/"+sha+".pt"}
                              for sha,entry in evidence_checkpoints(root).items()],"feature_caches":caches}
        index["index_id"]=digest(index)
        return index

    def test_index_requires_every_exact_retained_milestone(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            index=self.fixture(root)
            self.assertEqual(validate_index(index,root),{"checkpoints":2,"feature_caches":3,"bytes":58})
            missing=copy.deepcopy(index)
            missing["checkpoints"].pop()
            missing["index_id"]=digest({k:v for k,v in missing.items() if k!="index_id"})
            with self.assertRaisesRegex(ValueError,"exact complete evidence"):
                validate_index(missing,root)
            missing=copy.deepcopy(index)
            missing["feature_caches"].pop()
            missing["index_id"]=digest({k:v for k,v in missing.items() if k!="index_id"})
            with self.assertRaisesRegex(ValueError,"three frozen feature caches"):
                validate_index(missing,root)
            index["checkpoints"][0]["bytes"]=13
            with self.assertRaisesRegex(ValueError,"integrity mismatch"):
                validate_index(index,root)

    def test_materialization_links_bytes_and_refuses_a_conflicting_object(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=root/"original.pt"
            source.write_bytes(b"original immutable checkpoint")
            expected={"bytes":source.stat().st_size,"sha256":file_hash(source)}
            target=materialize(source,root/"objects",expected)
            self.assertEqual(target.stat().st_ino,source.stat().st_ino)
            self.assertEqual(materialize(source,root/"objects",expected),target)
            other=root/"conflict"; other.mkdir()
            bad=other/target.name; bad.write_bytes(b"unrelated data")
            with self.assertRaises(ValueError):
                materialize(source,other,expected)
            self.assertEqual(bad.read_bytes(),b"unrelated data")
            self.assertEqual(file_hash(source),expected["sha256"])

    def test_shared_static_checkpoint_selection_and_missing_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); index=self.fixture(root)
            write_json(root/"checkpoints.json",index)
            options=dict(host="qwen08",cohort="U",modality="vision",kind="static",
                         feature_mode="true",seed=42,step=3000,diversity=0.)
            with patch("legolm_p01.checkpoints.download_verified") as download:
                result=fetch(root/"checkpoints.json",root/"downloads",**options)
                self.assertEqual(result["sha256"],"b"*64)
                self.assertEqual(download.call_count,1)
                with self.assertRaisesRegex(ValueError,"absent or ambiguous"):
                    fetch(root/"checkpoints.json",root/"downloads",**{**options,"seed":1042})
                self.assertEqual(download.call_count,1)


if __name__=="__main__":
    unittest.main()
