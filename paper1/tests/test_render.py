import json
from pathlib import Path
import tempfile
import unittest

from legolm_p01.render import Report, validate_summary


class ManuscriptIntegrity(unittest.TestCase):
    def test_incomplete_summary_requires_explicit_draft(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            path.write_text(json.dumps({"schema": "paper1-summary-v2", "status": "incomplete"}))
            with self.assertRaisesRegex(ValueError, "complete, locally verifiable"):
                validate_summary(path, None)
            self.assertEqual(validate_summary(path, None, draft=True)["status"], "incomplete")

    def test_raw_generation_uses_frozen_p_prompt_reference(self):
        condition = {"host": "qwen08", "modality": "imu", "cohort": "U", "kind": "none",
                     "feature_mode": "true", "diversity": 0., "step": 0, "measurement": "generation",
                     "settings": {"objective": "P", "intervention": "correct"}}
        value = {"mean": .25, "ci95": [.1, .4]}
        row = {"condition_id": "frozen-raw", "condition": condition,
               "statistics": {"label/original-ranking/joint_success": {"macro": value, "micro": {"mean": .3}}}}
        with tempfile.TemporaryDirectory() as temporary:
            report = Report({"conditions": [row]}, temporary)
            self.assertEqual(report.stat("qwen08", "imu", "generation", "label/original-ranking/joint_success", "macro",
                                         cohort="P", kind="none"), value)
            with self.assertRaisesRegex(ValueError, "Missing or ambiguous"):
                report.find("qwen08", "imu", "generation", cohort="G", kind="none")


if __name__ == "__main__":
    unittest.main()
