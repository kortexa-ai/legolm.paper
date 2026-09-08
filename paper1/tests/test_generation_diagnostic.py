import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from legolm_p01.generation_diagnostic import observe, run
from legolm_p01.provenance import digest


class DiagnosticRMSNorm(torch.nn.Module):
    def forward(self, value):
        return value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + 1e-6)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.norm = DiagnosticRMSNorm()

    def forward(self, value):
        return SimpleNamespace(logits=self.model.norm(value))


class GenerationDiagnosticTests(unittest.TestCase):
    def host(self):
        return SimpleNamespace(model=TinyModel(), tokens=SimpleNamespace(decode=lambda ids: "!"))

    def test_finite_inputs_can_overflow_variance_and_produce_zero_logits(self):
        host = self.host()
        def decode(*args, **kwargs):
            host.model(torch.full((1, 2, 3), 1e20))
            host.model(torch.full((1, 1, 3), 1e20))
            return ["!!"]
        with patch("legolm_p01.generation_diagnostic.decode", side_effect=decode):
            result = observe(host, None, None, ["test"])
        norm = result["normalizations"][0]
        self.assertEqual(norm["input"]["nonfinite"], 0)
        self.assertEqual(norm["fp32_variance_nonfinite"], 2)
        self.assertGreater(norm["max_fp64_variance"], 1e39)
        self.assertEqual(norm["output"]["zeros"], 6)
        self.assertEqual(len(result["normalizations"]), 1)
        self.assertEqual(len(result["steps"]), 2)
        self.assertEqual(result["steps"][0]["rows"][0]["zeros"], 3)
        self.assertEqual(result["responses"], ["!!"])
        self.assertTrue(all(not m._forward_hooks for m in host.model.modules()))

    def test_failed_decoding_removes_hooks(self):
        host = self.host()
        with patch("legolm_p01.generation_diagnostic.decode", side_effect=RuntimeError("decode failed")):
            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                observe(host, None, None, ["test"])
        self.assertTrue(all(not m._forward_hooks for m in host.model.modules()))

    def test_existing_diagnostic_is_never_replaced(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            output.write_text("preserved")
            with self.assertRaises(FileExistsError):
                run("missing", "missing", directory, directory, directory, output, "cpu")
            self.assertEqual(output.read_text(), "preserved")

    def test_mismatched_reference_fails_before_tensor_loading(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, reference = root / "checkpoint.pt", root / "reference.json"
            checkpoint.write_bytes(b"not a tensor container")
            identity = {"checkpoint_sha256": "0" * 64}
            reference.write_text(json.dumps({"identity": identity, "evaluation_id": digest(identity)}))
            with self.assertRaisesRegex(ValueError, "another checkpoint"):
                run(checkpoint, reference, root, root, root, root / "new.json", "cpu")


if __name__ == "__main__":
    unittest.main()
