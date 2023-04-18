## SnapCCESS

A R package wrapper of python package SnapCCESS to generate ensemble deep learning of embeddings for clustering multimodal single-cell omics data




## Installation

### Stable version
```
remotes::install_github(repo='PYangLab/SnapCCESS',branch='main',subdir='snapccess-r/SnapCCESS')
``` 
 

### Development version (not recommanded)
```
remotes::install_github(repo='yulijia/SnapCCESS',branch='main',subdir='snapccess-r/SnapCCESS') 
``` 
 

## The functions in this package are described below.

### install_SnapCCESS

#### Description

install related python packages, this should be run only once when you install the R pacakge.



#### Usage

```
install_SnapCCESS(envname = "SnapCCESS",method = "conda",conda="auto")
```

#### Arguments


- `envname`: 
The name, or full path, of the environment in which Python packages are to be installed. When NULL (the default), the active environment as set by the RETICULATE_PYTHON_ENV variable will be used; if that is unset, then the r-reticulate environment will be used.

- `method`:	
Installation method. By default, "auto" automatically finds a method that will work in the local environment. Change the default to force a specific installation method. Note that the "virtualenv" method is not available on Windows.

- `conda`:	
The path to a conda executable. By default, reticulate will check the PATH, as well as other standard locations for Anaconda installations.

- `...`:	
Additional arguments passed to reticulate::py_install function
 


----------------------

### loadmodule

#### Description

Load all python modules that may used during analysis automatically, including torch, snapccess, and numpy

#### Usage

```
loadmodule()
```

#### Arguments
 
No arguments are needed

-------------------

### preprocess

#### Description

Data preprocess, PyTorch data loading utility

#### Usage

```
preprocess(x, mb_size = 64, num_workers = 0)
```


#### Arguments

- `x`:	
a list of input data modalities

- `mb_size`:	
mini batch size

- `num_workers`:	
Setting the argument num_workers as a positive integer will turn on multi-process data loading with the specified number of loader worker processes.

#### Output

a python object loaded datasets


-------------------

### build_model

#### Description

Build the SnapCCESS VAE model

#### Usage

```
build_model(num_features, num_hidden_features, num_latent_features)
```

#### Arguments

- `num_features`: 
number of input features

- `num_hidden_features`:	
number of hidden layer features

- `num_latent_features`:	
number of the features in embeddings



#### Output

a python object of deep learning model


-------------------

### run_SnapCCESS

#### Description

Run the SnapCCESS or Traditional VAE trainning process to get the embeddings.

#### Usage

```
output=run_SnapCCESS(model,data,epochs=50,epochs_per_cycle=2,save_path="",snapshot=TRUE)
```

#### Arguments


- `model`:	
SnapCCESS VAE model

- `data`:	
preprocessed dataset

- `lr`:	
initial learning rate

- `epochs`:	
number of epochs cycle, if snapshot is false, this become to the total epochs that will be training in traditional VAE model

- `epochs_per_cycle`:
number of epochs per cycle, if snapshot is false, this argument doesn't work

- `save_path`:	
which specifies the folder where the embedding will be saved. If save_path is not specified or is an empty string, the embedding will only be returned as an R object and will not be saved to the local folder.

- `snapshot`:	
Boolen value, True for run SnapCCESS, False will run traditional VAE

- `embedding_number`:	
the number in the filename to indicate that which embedding is saved to folder.



#### Output

a list of python objects, including model, loss of each epoch, and embeddings
