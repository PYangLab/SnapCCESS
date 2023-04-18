# SnapCCESS
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


For python version of script, please see [an_example_of_generate_embedding_using_SnapCCESS_python_version](https://github.com/PYangLab/SnapCCESS/blob/main/tutorials/src/an_example_of_generate_embedding_using_SnapCCESS_python_version.ipynb)

For R version of script, please see
[SnapCCESS_R_example](https://htmlpreview.github.io/?https://github.com/PYangLab/SnapCCESS/blob/main/tutorials/src/SnapCCESS_R_example.html)

