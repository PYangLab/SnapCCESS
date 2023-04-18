## SnapCCESS

A python package to generate ensemble deep learning of embeddings for clustering multimodal single-cell omics data


## Installation

### Stable version
```
pip install snapccess  --index-url https://pypi.org/simple
``` 

https://pypi.org/project/snapccess/


### Development version
```
pip install snapccess  --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple
``` 

https://test.pypi.org/project/snapccess/


## The functions in this package are described below.

### snapshotVAE

#### Description

To create the VAE model

#### Usage

```
model = snapshotVAE(num_features=[nfeatures_rna,nfeatures_pro], num_hidden_features=[hidden_rna2,hidden_pro2], z_dim=z_dim)
```

#### Arguments

- num_features: a list of number of features of each modality
- num_hidden_features: the number of hidden features we will used in training the model, in our paper, we use `hidden_rna=185`, and `hidden_pro=30`
- z_dim: dimension of the latent space, in our paper, we use `z_dim=100`
 

#### Output

A VAE model


----------------------

### train_model

#### Description

Training a VAE model with Snapshot learning rate or constant learning rate 


#### Usage

```
model,histroy,embedding = train_model(model, train_dl, valid_dl, lr=lr, epochs=epochs,epochs_per_cycle=epochs_per_cycle, save_path=\"\",snapshot=True,embedding_number=1)
```

#### Arguments

- model: a vae model
- train_dl: training dataset
- valid_dl: validation dataset
- lr: initial learning rate
- epochs: total number of train cycles for snapshot ensemble vae
- epochs_per_cycle: the number of epochs per cycle
- save_path: the output file path of embeddings, by default leave it blank will not save any embeddings into the `save_path`, but the `train_model` will return the embeddings
- snapshot: a boolean value to indicate the model whether to use the snapshot ensemble method or the traditional VAE method (with constant learning rate)
- embeddings_number: a value to indicate the index of embeddings in the output filename when apply the traditional VAE


#### Output

This function will return the model, the loss of training and validation dataset (history) and a list of the latent space embeddings (for Snapshot ensemble method) or a single embedding for traditional VAE method.


-------------------

### get_encodings

#### Description

To get the embeddings from model after training.

#### Usage

```
embedding = get_encodings(model,valid_dl)
```


#### Arguments

- model: a VAE model
- valid_dl: the dataset that used as input to training the VAE model

#### Output

Embedding of the `valid_dl` dataset in the VAE model, to convert it to a matrix, try `pd.DataFrame(embedding.cpu().numpy())`


-------------------

### nvidia_info

#### Description

To monitor the memory usage of GPU

#### Usage

```
memory = nvidia_info(pid)['memory']
```

#### Arguments

- pid: the pid of running script


#### Output

This function will return the memory usage of the pid process.