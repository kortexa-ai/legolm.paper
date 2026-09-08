from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from legolm_p01.portable_text import content_identity, load_verified_text
from legolm_p01.provenance import file_hash, write_json


class PortableTextIntegrity(unittest.TestCase):
    def fixture(self, root):
        cache = {"identity": {"host": "mini", "tokenizer": "fixed"},
                 "train": torch.arange(12).reshape(3, 4), "val": torch.arange(4).reshape(1, 4)}
        path = root / "tokens.pt"
        torch.save(cache, path)
        write_json(root / "text-mini.json", {**cache["identity"], "file": path.name, "sha256": file_hash(path)})
        reference = root / "reference.json"
        write_json(reference, {"hosts": {"mini": {"file_sha256": "a" * 64, **content_identity(cache)}}})
        return cache, path, reference

    def test_exact_file_and_identical_tokens_are_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, path, reference = self.fixture(root)
            _, exact = load_verified_text(root, "mini", file_hash(path))
            self.assertEqual(exact["mode"], "identical_file")
            with patch("legolm_p01.portable_text.reference_path", return_value=reference):
                regenerated, verified = load_verified_text(root, "mini", "a" * 64)
            self.assertEqual(verified["mode"], "identical_token_arrays")
            self.assertTrue(torch.equal(regenerated["train"], cache["train"]))

    def test_changed_token_is_rejected_even_with_valid_local_file_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, path, reference = self.fixture(root)
            cache["val"][0, 0] += 1
            torch.save(cache, path)
            write_json(root / "text-mini.json", {**cache["identity"], "file": path.name, "sha256": file_hash(path)})
            with patch("legolm_p01.portable_text.reference_path", return_value=reference):
                with self.assertRaisesRegex(ValueError, "token arrays or input identity changed"):
                    load_verified_text(root, "mini", "a" * 64)

    def test_unregistered_checkpoint_or_changed_input_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, path, reference = self.fixture(root)
            with patch("legolm_p01.portable_text.reference_path", return_value=reference):
                with self.assertRaisesRegex(ValueError, "No registered"):
                    load_verified_text(root, "mini", "b" * 64)
                cache["identity"]["tokenizer"] = "different"
                torch.save(cache, path)
                write_json(root / "text-mini.json", {**cache["identity"], "file": path.name, "sha256": file_hash(path)})
                with self.assertRaisesRegex(ValueError, "token arrays or input identity changed"):
                    load_verified_text(root, "mini", "a" * 64)


if __name__ == "__main__":
    unittest.main()
