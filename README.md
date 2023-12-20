# SnapCCESS <a href="https://github.com/PYangLab/SnapCCESS"><img src="https://i.imgur.com/XHEB9j1.png" title="SnapCCESS hex sticker" align="right" height="138" /></a>

SnapCCESS: Ensemble deep learning of embeddings for clustering multimodal single-cell omics data.


We propose SnapCCESS for clustering cells by integrating data modalities in multimodal
single-cell omics data using an unsupervised ensemble deep learning framework. By creating snapshots
of embeddings of multimodality using variational autoencoders, SnapCCESS can be coupled with
various clustering algorithms for generating consensus clustering of cells.


![img](https://i.imgur.com/krfBTGP.png)



## Installation

### Python

```
pip install snapccess  --index-url https://pypi.org/simple
```

For detailed description of each function, please see [https://github.com/PYangLab/SnapCCESS/tree/main/snapccess-py](https://github.com/PYangLab/SnapCCESS/tree/main/snapccess-py)


### R

```
remotes::install_github(repo='PYangLab/SnapCCESS',branch='main',subdir='snapccess-r/SnapCCESS')
```

For detailed description of each function, please see [https://github.com/PYangLab/SnapCCESS/tree/main/snapccess-r](https://github.com/PYangLab/SnapCCESS/tree/main/snapccess-r)


## [Tutorial](https://github.com/PYangLab/SnapCCESS/tree/main/tutorials)
### NOTE: This tutorial only explains how to use this package; it doesn't recommend the best parameters for your datasets. For the datasets used in the published paper associated with this package, the parameters are listed in the same paper. Please refer to the paper to guide you in finding the best parameters.


For python version of script, please see [an_example_of_generate_embedding_using_SnapCCESS_python_version](https://github.com/PYangLab/SnapCCESS/blob/main/tutorials/src/an_example_of_generate_embedding_using_SnapCCESS_python_version.ipynb)

For R version of script, please see
[SnapCCESS_R_example](https://htmlpreview.github.io/?https://github.com/PYangLab/SnapCCESS/blob/main/tutorials/src/SnapCCESS_R_example.html)

## References
Lijia Yu, Chunlei Liu, Jean Yee Hwa Yang, Pengyi Yang. Ensemble deep learning of embeddings for clustering multimodal single-cell omics data. *Bioinformatics*, 39(6), btad382, doi: [https://doi.org/10.1093/bioinformatics/btad382](https://doi.org/10.1093/bioinformatics/btad382), (2023).
