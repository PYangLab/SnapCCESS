# snapccess

Python implementation of SnapCCESS, an ensemble deep learning framework for
learning multimodal single-cell embeddings for downstream clustering.

## Installation

```bash
pip install snapccess --index-url https://pypi.org/simple
```

Development builds, when available, can be installed from TestPyPI:

```bash
pip install snapccess --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple
```

## Main Functions

### `snapshotVAE`

Create the multimodal variational autoencoder model.

```python
from snapccess.model import snapshotVAE

model = snapshotVAE(
    num_features=[nfeatures_rna, nfeatures_protein],
    num_hidden_features=[hidden_rna, hidden_protein],
    z_dim=100,
)
```

### `train_model`

Train a VAE with either snapshot learning-rate cycles or a constant learning
rate.

```python
from snapccess.train import train_model

model, history, embeddings = train_model(
    model,
    train_dl,
    valid_dl,
    lr=0.02,
    epochs=50,
    epochs_per_cycle=2,
    save_path="",
    snapshot=True,
)
```

The function returns the trained model, training/validation loss history, and a
list of latent-space embeddings. When `save_path` is provided, embeddings are
also written as gzip-compressed CSV files.

### `get_encodings`

Return embeddings from a trained model.

```python
from snapccess.util import get_encodings

embedding = get_encodings(model, valid_dl)
```

Convert the result to a pandas data frame with:

```python
import pandas as pd

embedding_df = pd.DataFrame(embedding.cpu().numpy())
```

### `nvidia_info`

Monitor GPU memory usage for a process when NVIDIA drivers and `pynvml` are
available.

```python
from snapccess.util import nvidia_info

memory = nvidia_info(pid)["memory"]
```

## Citation

If you use SnapCCESS, please cite:

Yu, L., Liu, C., Yang, J. Y. H. & Yang, P. Ensemble deep learning of embeddings
for clustering multimodal single-cell omics data. *Bioinformatics* 39(6),
btad382 (2023). <https://doi.org/10.1093/bioinformatics/btad382>
