import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


try:
    import torch
    from snapccess.model import snapshotVAE
    from snapccess.train import train_model
except ModuleNotFoundError as exc:
    torch = None
    snapshotVAE = None
    train_model = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(torch is None, f"snapccess dependencies are not installed: {IMPORT_ERROR}")
class ModelTrainTest(unittest.TestCase):
    def test_snapshot_vae_forward_shapes(self):
        model = snapshotVAE(num_features=[3, 2], num_hidden_features=[4, 3], z_dim=2)
        x = torch.randn(5, 5)

        decoded, mu, var = model(x)

        self.assertEqual(tuple(decoded.shape), (5, 5))
        self.assertEqual(tuple(mu.shape), (5, 2))
        self.assertEqual(tuple(var.shape), (5, 2))

    def test_train_model_records_validation_loss_and_embedding(self):
        torch.manual_seed(1)
        model = snapshotVAE(num_features=[3, 2], num_hidden_features=[4, 3], z_dim=2)
        data = torch.randn(8, 5)
        loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=False)

        _, history, embeddings = train_model(
            model,
            loader,
            loader,
            lr=0.001,
            epochs=1,
            epochs_per_cycle=1,
            verbose=False,
            snapshot=False,
        )

        self.assertEqual(len(history["train"]), 1)
        self.assertEqual(len(history["valid"]), 1)
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(tuple(embeddings[0].shape), (8, 2))


if __name__ == "__main__":
    unittest.main()
