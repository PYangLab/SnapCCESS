# SnapCCESS <a href="https://github.com/PYangLab/SnapCCESS"><img src="https://i.imgur.com/XHEB9j1.png" title="SnapCCESS hex sticker" align="right" height="138" /></a>

SnapCCESS is an unsupervised ensemble deep learning framework for clustering
multimodal single-cell omics data. It creates snapshots of multimodal variational
autoencoder embeddings, which can then be used with downstream clustering
methods to generate consensus cell clusters.

The repository contains both:

- `snapccess-py/`: the Python implementation distributed through PyPI.
- `snapccess-r/SnapCCESS/`: an R wrapper that calls the Python package through
  `reticulate`.

![SnapCCESS workflow](https://i.imgur.com/krfBTGP.png)

## Publication

SnapCCESS accompanies:

> Yu, L., Liu, C., Yang, J. Y. H. & Yang, P. Ensemble deep learning of
> embeddings for clustering multimodal single-cell omics data.
> *Bioinformatics* 39(6), btad382 (2023).
> <https://doi.org/10.1093/bioinformatics/btad382>

## Installation

### Python

```bash
pip install snapccess --index-url https://pypi.org/simple
```

The Python package depends on PyTorch. GPU acceleration is optional and depends
on the local PyTorch/CUDA or PyTorch/ROCm installation.

### R Wrapper

```r
remotes::install_github(
  repo = "PYangLab/SnapCCESS",
  branch = "main",
  subdir = "snapccess-r/SnapCCESS"
)
```

After installing the R wrapper, install the Python package into a reticulate
environment:

```r
SnapCCESS::install_SnapCCESS(envname = "SnapCCESS", method = "conda")
```

If Python dependencies are already managed through Conda, virtualenv, or a
cluster module, configure `reticulate` to use that environment before loading
the R wrapper.

## Input Data

SnapCCESS expects a list of modalities measured across the same cells, for
example RNA and ADT matrices from CITE-seq. Each modality should be arranged as
features by cells before preprocessing in the R wrapper. The Python API expects
the combined PyTorch data loader used by the training routine.

The tutorial data in `tutorials/in/` includes small CITE-seq RNA, ADT, and cell
type files that can be used to run the examples.

## Tutorials

- Python example:
  [`tutorials/src/an_example_of_generate_embedding_using_SnapCCESS_python_version.ipynb`](tutorials/src/an_example_of_generate_embedding_using_SnapCCESS_python_version.ipynb)
- R example:
  [`tutorials/src/SnapCCESS_R_example.html`](tutorials/src/SnapCCESS_R_example.html)
- Tutorial overview:
  [`tutorials/README.md`](tutorials/README.md)

The tutorials demonstrate package usage. The best hyperparameters can vary by
dataset; for the datasets used in the publication, refer to the paper.

## Citation

If you use SnapCCESS, please cite:

```bibtex
@article{Yu2023SnapCCESS,
  title = {Ensemble deep learning of embeddings for clustering multimodal single-cell omics data},
  author = {Yu, Lijia and Liu, Chunlei and Yang, Jean Yee Hwa and Yang, Pengyi},
  journal = {Bioinformatics},
  year = {2023},
  volume = {39},
  number = {6},
  pages = {btad382},
  doi = {10.1093/bioinformatics/btad382},
  url = {https://doi.org/10.1093/bioinformatics/btad382}
}
```

## License

SnapCCESS is distributed under the GPL-3 license.
