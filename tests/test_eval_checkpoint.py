import os
import tempfile
import unittest

import torch
import torch.nn as nn

import eval


class FakeDataParallel:
    """Minimal stand-in for torch.nn.DataParallel that load_checkpoint expects."""
    def __init__(self, module):
        self.module = module


class TestPartialCheckpointLoading(unittest.TestCase):
    def test_shape_mismatch_keys_are_skipped(self):
        net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.Conv2d(8, 16, 3, padding=1),
        )
        model = FakeDataParallel(net)

        original_conv0 = net[0].weight.data.clone()
        original_conv1 = net[1].weight.data.clone()

        # Checkpoint: first layer matches, second layer has wrong shape, plus an extra key.
        checkpoint = {
            'epoch': 7,
            'state_dict': {
                '0.weight': original_conv0.clone() * 2.0,
                '1.weight': torch.zeros(32, 8, 3, 3),  # mismatch vs. expected (16, 8, 3, 3)
                'extra_key': torch.zeros(1, 1),
            },
        }

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(checkpoint, f.name)
            path = f.name

        try:
            epoch = eval.load_checkpoint(model, path)
        finally:
            os.remove(path)

        self.assertEqual(epoch, 7)
        # Matching key should be loaded from checkpoint.
        self.assertTrue(torch.allclose(net[0].weight.data, original_conv0 * 2.0))
        # Mismatched-shape key should be skipped, leaving original values.
        self.assertTrue(torch.allclose(net[1].weight.data, original_conv1))

