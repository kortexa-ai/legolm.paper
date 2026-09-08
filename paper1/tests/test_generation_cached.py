import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from legolm_p01 import generation
from legolm_p01.generation_cached import decode, evaluate_generation


class GenerationBatchReuse(unittest.TestCase):
    def setUp(self):
        self.data = {"modality": "audio", "categories": ["cat", "dog"], "prompt": "The sound is: ",
                     "test": {"ids": [str(i) for i in range(400)], "labels": [i % 2 for i in range(400)],
                              "groups": ["source-" + str(i // 2) for i in range(400)],
                              "features": torch.arange(1600, dtype=torch.float32).view(400, 4)}}
        self.constant = torch.ones(4)
        self.host = SimpleNamespace(name="qwen08", tokens=SimpleNamespace(hf=SimpleNamespace(eos_token_id=248046)),
                                    model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=248044)))

    @staticmethod
    def decoder(host, adapter, kind, features, prompts, *, native_chat, intervention):
        # A deterministic, input-dependent decoder fixture. Batch order and
        # label-in-text prompts matter; variable sensors must not share output.
        sensor_free = kind in ("none", "static") or intervention == "off"
        return [str(sum(map(ord, prompt)) + (0 if sensor_free else int(feature.sum())))
                for prompt, feature in zip(prompts, features)]

    def test_fixed_batches_preserve_all_6400_records_and_label_prompts(self):
        for kind, intervention, oracle in (("none", "correct", None), ("static", "correct", None),
                                           ("bridge", "off", None), ("bridge", "constant", None),
                                           ("none", "correct", self.data["test"]["labels"])):
            with self.subTest(kind=kind, intervention=intervention, oracle=oracle is not None):
                args = (self.host, None, kind, self.data, self.constant)
                kwargs = dict(objective="G", intervention=intervention, batch_size=8, oracle_labels=oracle)
                with patch.object(generation, "decode", side_effect=self.decoder):
                    reference = generation.evaluate_generation(*args, **kwargs)
                with patch.object(generation, "decode", side_effect=self.decoder) as decoder:
                    result = evaluate_generation(*args, **kwargs)
                    self.assertLess(decoder.call_count, 40)
                execution = result.pop("execution")
                self.assertEqual(result["decoding"].pop("eos_token_ids"), [248044, 248046])
                self.assertEqual(execution["batch_size"], 8)
                self.assertGreater(execution["reused_identical_batches"], 750)
                self.assertEqual(len(result["items"]), 6400)
                self.assertEqual(result, reference)

    def test_true_sensor_batches_are_never_reused(self):
        args = (None, None, "bridge", self.data, self.constant)
        with patch.object(generation, "decode", side_effect=self.decoder):
            reference = generation.evaluate_generation(*args, objective="P", batch_size=8)
        with patch.object(generation, "decode", side_effect=self.decoder) as decoder:
            result = evaluate_generation(*args, objective="P", batch_size=8)
            self.assertEqual(decoder.call_count, 50)
        self.assertEqual(result.pop("execution")["reused_identical_batches"], 0)
        self.assertEqual(result, reference)

    def test_native_chat_stops_at_assistant_end_and_restores_configuration(self):
        original = self.host.model.generation_config
        def observed(*args, **kwargs):
            self.assertEqual(self.host.model.generation_config.eos_token_id, [248044, 248046])
            self.assertIsNot(self.host.model.generation_config, original)
            return ["walking"]
        with patch.object(generation, "decode", side_effect=observed):
            self.assertEqual(decode(self.host, None, "none", self.constant, ["prompt"], native_chat=True,
                                    intervention="correct"), ["walking"])
        self.assertIs(self.host.model.generation_config, original)
        self.assertEqual(original.eos_token_id, 248044)

    def test_native_chat_restores_configuration_after_decode_error(self):
        original = self.host.model.generation_config
        with patch.object(generation, "decode", side_effect=RuntimeError("decoder failed")):
            with self.assertRaisesRegex(RuntimeError, "decoder failed"):
                decode(self.host, None, "none", self.constant, ["prompt"], native_chat=True, intervention="correct")
        self.assertIs(self.host.model.generation_config, original)

    def test_raw_generation_retains_the_model_stopping_policy(self):
        original = self.host.model.generation_config
        def observed(*args, **kwargs):
            self.assertFalse(kwargs["native_chat"])
            self.assertIs(self.host.model.generation_config, original)
            return ["walking"]
        with patch.object(generation, "decode", side_effect=observed):
            decode(self.host, None, "none", self.constant, ["prompt"], native_chat=False, intervention="correct")


if __name__ == "__main__":
    unittest.main()
