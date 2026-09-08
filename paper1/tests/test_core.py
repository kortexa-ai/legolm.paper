import unittest
from unittest.mock import Mock

import torch
from torch import nn

from legolm_p01.hosts import LoRA, Bridge
from legolm_p01.text import label_rows


class ScientificContracts(unittest.TestCase):
    def test_generated_updates_and_per_example_isolation(self):
        torch.manual_seed(7)
        base = nn.Linear(5, 7)
        layer = LoRA(base, 2)
        x = torch.randn(3, 4, 5)
        a, b = torch.randn(3, 5, 2), torch.randn(3, 2, 7)
        layer.generated = (a, b)
        expected = torch.stack([base(x[i]) + x[i] @ a[i] @ b[i] for i in range(3)])
        torch.testing.assert_close(layer(x), expected)
        changed = a.clone()
        changed[1] += 1
        layer.generated = (changed, b)
        torch.testing.assert_close(layer(x)[[0, 2]], expected[[0, 2]])
        self.assertFalse(torch.allclose(layer(x)[1], expected[1]))

    def test_candidate_groups_reuse_only_their_own_sensor(self):
        layer = LoRA(nn.Linear(3, 2), 1)
        a, b = torch.randn(2, 3, 1), torch.randn(2, 1, 2)
        x = torch.randn(2, 4, 3)
        layer.generated = (a,b)
        direct = layer(x)
        layer.batch_repeat = 3
        repeated = layer(x.repeat_interleave(3,dim=0))
        torch.testing.assert_close(repeated,direct.repeat_interleave(3,dim=0))

    def test_effective_sum_is_not_factor_average(self):
        layer = LoRA(nn.Linear(2, 2, bias=False), 1)
        x = torch.eye(2)[None]
        a1, a2 = torch.tensor([[1.], [0.]]), torch.tensor([[0.], [1.]])
        b1, b2 = a1.T, a2.T
        layer.parts = [(a1, b1), (a2, b2)]
        delta = layer(x) - layer.base(x)
        torch.testing.assert_close(delta[0], torch.eye(2))
        layer.parts = None
        layer.generated = ((a1+a2)/2, (b1+b2)/2)
        self.assertFalse(torch.allclose(layer(x)-layer.base(x), delta/2))

    def test_gradient_reaches_bridge_only(self):
        layer = LoRA(nn.Linear(3, 4), 2)
        bridge = Bridge(5, 14)
        features, x = torch.randn(2, 5), torch.randn(2, 3, 3)
        generated = bridge(features)
        layer.generated = (generated[:, :6].view(2, 3, 2), generated[:, 6:].view(2, 2, 4))
        layer(x).square().mean().backward()
        self.assertGreater(bridge.mlp[-1].weight.grad.norm().item(), 0)
        self.assertIsNone(layer.base.weight.grad)
        self.assertIsNone(layer.a.grad)

    def test_label_boundary_merge_and_no_double_shift(self):
        tokens = Mock()
        tokens.task_encode.side_effect = lambda s: {"P ": [1,2,3], "P a": [1,2,4], "P bc": [1,2,5,6]}[s]
        x,y,mask = label_rows(tokens, "P ", ["a", "bc"])
        self.assertEqual(y[0][mask[0]].tolist(), [4])
        self.assertEqual(y[1][mask[1]].tolist(), [5,6])
        self.assertEqual(x[0].tolist(), [1,2,0])


if __name__ == "__main__":
    unittest.main()
