import tempfile
from pathlib import Path
import unittest

from legolm_p01.artifacts import measurement_key
from legolm_p01.provenance import digest, file_hash, write_json
from legolm_p01.reconcile import select_catalog


class EvaluatorReconciliation(unittest.TestCase):
    def test_selects_current_source_even_when_old_score_is_better(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, backend = {"evaluation_files": {"scorer.py": "new"}}, {"device": "fixed"}
            catalog = {}
            for revision, score in (("old", .1), ("new", 9.)):
                identity = {"evaluation_source": {"evaluation_files": {"scorer.py": revision}},
                            "backend": backend, "sensor_batch": 4, "measurement": "native_bpb", "rows": 1024}
                key = digest(identity)
                path = root / (key + ".json")
                write_json(path, {"identity": identity, "summary": {"bpb": score}})
                catalog[key] = {"evaluation_id": key, "path": path.name, "sha256": file_hash(path),
                                "measurement": "native_bpb", "settings": {"rows": 1024}, "summary": {"bpb": score}}
            write_json(root / "evaluations.json", catalog)
            original_hashes = {p.name: file_hash(p) for p in root.iterdir()}
            expected = {measurement_key("native_bpb", {"rows": 1024}): {}}
            _, retained, retired, missing = select_catalog(root, expected, source, backend)
            self.assertFalse(missing)
            self.assertEqual([r["summary"]["bpb"] for r in retained.values()], [9.])
            self.assertEqual([r["summary"]["bpb"] for r in retired.values()], [.1])
            self.assertEqual(original_hashes, {p.name: file_hash(p) for p in root.iterdir()})

    def test_wrong_backend_or_batch_requires_rescoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, backend = {"evaluation_files": {"scorer.py": "new"}}, {"device": "fixed"}
            identity = {"evaluation_source": source, "backend": backend, "sensor_batch": 1}
            key = digest(identity)
            path = root / (key + ".json")
            write_json(path, {"identity": identity})
            write_json(root / "evaluations.json", {key: {"evaluation_id": key, "path": path.name,
                "sha256": file_hash(path), "measurement": "common_bpb", "settings": {}}})
            expected = {measurement_key("common_bpb", {}): {}}
            _, retained, retired, missing = select_catalog(root, expected, source, backend)
            self.assertFalse(retained)
            self.assertEqual(len(retired), 1)
            self.assertEqual(missing, set(expected))


if __name__ == "__main__":
    unittest.main()
