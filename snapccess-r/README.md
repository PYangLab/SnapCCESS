# SnapCCESS R Wrapper

The R package wraps the Python `snapccess` package through `reticulate`, allowing
R users to preprocess multimodal single-cell matrices, build the SnapCCESS VAE,
and train snapshot embeddings.

## Installation

```r
remotes::install_github(
  repo = "PYangLab/SnapCCESS",
  branch = "main",
  subdir = "snapccess-r/SnapCCESS"
)
```

Install the Python dependency into a reticulate environment:

```r
SnapCCESS::install_SnapCCESS(envname = "SnapCCESS", method = "conda")
```

## Main Functions

| Function | Purpose |
| --- | --- |
| `install_SnapCCESS()` | Install the Python package into a reticulate environment. |
| `loadmodule()` | Load Python modules used by the wrapper. |
| `preprocess()` | Scale and combine input modalities into a PyTorch data loader. |
| `build_model()` | Build the SnapCCESS VAE model. |
| `run_SnapCCESS()` | Train the model and return snapshot embeddings. |

## Minimal Workflow

```r
library(SnapCCESS)

loadmodule()

data <- preprocess(list(rna, adt), mb_size = 64)

model <- build_model(
  num_features = list(nrow(rna), nrow(adt)),
  num_hidden_features = list(185, 30),
  num_latent_features = 100
)

output <- run_SnapCCESS(
  model,
  data,
  epochs = 50,
  epochs_per_cycle = 2,
  save_path = "",
  snapshot = TRUE
)
```

## Citation

If you use SnapCCESS, please cite:

Yu, L., Liu, C., Yang, J. Y. H. & Yang, P. Ensemble deep learning of embeddings
for clustering multimodal single-cell omics data. *Bioinformatics* 39(6),
btad382 (2023). <https://doi.org/10.1093/bioinformatics/btad382>
