import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


try:
    from snapccess.util import snapshot_lr
except ModuleNotFoundError as exc:
    snapshot_lr = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(snapshot_lr is None, f"snapccess dependencies are not installed: {IMPORT_ERROR}")
class SnapshotLearningRateTest(unittest.TestCase):
    def test_snapshot_lr_starts_at_initial_learning_rate(self):
        self.assertAlmostEqual(snapshot_lr(0.02, epoch=1, epoch_per_cycle=4), 0.02)

    def test_snapshot_lr_follows_cosine_schedule(self):
        observed = snapshot_lr(0.02, epoch=3, epoch_per_cycle=4)
        expected = 0.02 * (math.cos(math.pi * 2 / 4) + 1) / 2
        self.assertAlmostEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
