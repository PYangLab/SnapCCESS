#' preprocess
#'
#' Data preprocess, PyTorch data loading utility
#'
#' @param x a list of input data modalities
#' @param mb_size mini batch size
#' @param num_workers Setting the argument num_workers as a positive integer will turn on
#' multi-process data loading with the specified number of loader worker processes.
#'
#' @return a python object loaded datasets
#' @export
#'
#' @examples #data=preprocess(lsit(rna,adt),mb_size=64)
#' @import reticulate
#'
#'
#'
preprocess=function(x,mb_size=64,num_workers=0){
  x_scale_t=lapply(x,function(x)t(scale(x)))

  seqdata=do.call(cbind,x_scale_t)
  train_data=reticulate::r_to_py(seqdata)
  train=train_data$astype(numpy$float32)

  train_load=torch$utils$data$DataLoader(train,batch_size=as.integer(mb_size),
                                         shuffle=FALSE,
                                         num_workers=as.integer(num_workers),
                                         drop_last=FALSE)
  return(train_load)
}


#' build_model
#'
#' Build the SnapCCESS VAE model
#'
#' @param num_features number of input features
#' @param num_hidden_features number of hidden layer features
#' @param num_latent_features number of the features in embeddings
#'
#' @return a python object of deep learning model
#' @export
#'
#' @import reticulate
#' @examples #model=build_model(num_features=list(1000,49),
#'           #num_hidden_features=list(185,30),
#'           #num_latent_features=100)
build_model=function(num_features,num_hidden_features,num_latent_features){
  nf=lapply(num_features,function(x)as.integer(x))
  nhf=lapply(num_hidden_features,function(x)as.integer(x))
  nlf=as.integer(num_latent_features)
  model=snapccess$model$snapshotVAE(num_features=r_to_py(num_features),
                                    num_hidden_features=r_to_py(nhf),
                                    z_dim=nlf
                                    )
  return(model)
}

#' run_SnapCCESS
#'
#' Run the SnapCCESS or Traditional VAE trainning process to get the embeddings.
#'
#' @param model SnapCCESS VAE model
#' @param data preprocessed dataset
#' @param lr initial learning rate
#' @param cycle number of epochs cycle, if snapshot is false,
#' this become to the total epochs that will be training in traditional VAE model
#' @param epochs_per_cycle number of epochs per cycle, if snapshot is false, this argument doesn't work
#' @param save_path which specifies the folder where the embedding will be saved.
#' If save_path is not specified or is an empty string, the embedding will only
#' be returned as an R object and will not be saved to the local folder.
#' @param snapshot Boolen value, True for run SnapCCESS, False will run traditional VAE
#' @param embedding_number the number in the filename to indicate that which embedding is saved to folder.
#'
#' @return a list of python objects, including model, loss of each epoch, and embeddings
#' @export
#' @import reticulate
#'
#' @examples #output=run_SnapCCESS(model,data,epochs=50,epochs_per_cycle=2,save_path="",snapshot=TRUE)
run_SnapCCESS=function(model,data,lr=0.02,cycle=50,epochs_per_cycle=1,
                       save_path="",snapshot=TRUE,embedding_number=1){
  res=snapccess$train$train_model(model, data, data, lr=lr,
                                  epochs=as.integer(cycle),
                                  epochs_per_cycle=as.integer(epochs_per_cycle),
                                  save_path=save_path,snapshot=snapshot,
                                  embedding_number=as.integer(embedding_number))
  return(res)
}
