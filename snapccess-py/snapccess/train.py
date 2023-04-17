import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
import os
import sys
import shutil
from snapccess.util import KL_loss,snapshot_lr,get_encodings
import pandas as pd

def train_model(model, train_dl: torch.utils.data.dataloader.DataLoader , valid_dl: torch.utils.data.dataloader.DataLoader , lr: float=0.02,  epochs: int=10, epochs_per_cycle: int=2, verbose: bool = True, save_path: str="", snapshot: bool=True, embedding_number: int=1):
    #####set optimizer and criterin#####
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) ##
    criterion = nn.MSELoss().to(device)
    
    history = defaultdict(list)

    ######loop training process, each epoch contains train and test two part#########
    
    embedding=[]
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        nsamples_train = 0
        train_loss=0
        model = model.train()
        if(snapshot):
            lr = snapshot_lr(lr, epoch, epochs_per_cycle)
            optimizer.state_dict()["param_groups"][0]["lr"] = lr
            
        for x in train_dl:
            optimizer.zero_grad()
            # Forward pass
            x_prime, mu, var = model(x.to(device))
            # loss function
            loss = criterion(x_prime, x.to(device)) + 0.0001*(KL_loss(mu,var))#simulation los x.to(device)
            # Backward pass
            loss.backward()
            optimizer.step()
            # log losses
            batch_size = x.shape[0]
            nsamples_train += batch_size
            train_loss += batch_size*(loss.item())

        valid_loss = 0
        nsamples_valid = 0
        model = model.eval()
        with torch.no_grad():
            for x in valid_dl:
                ###forward process
                x_prime, mu, var = model(x.to(device))
                
                # log losses
                batch_size = x.shape[0]
                nsamples_valid += batch_size
                valid_loss += batch_size*(loss.item())
        
        train_loss = train_loss / nsamples_train
        valid_loss = valid_loss / nsamples_valid

        history['train'].append(train_loss)
        history['valid'].append(valid_loss)
        
        if(snapshot):
            if (epoch!=0) and ((epoch+1)%epochs_per_cycle==0):
                simulated_data_ls = get_encodings(model,valid_dl)
                temp= pd.DataFrame(simulated_data_ls.cpu().numpy())
                embedding.append(temp)
                if save_path:
                    pd.DataFrame(simulated_data_ls.cpu().numpy()).to_csv(
                        save_path+'_embedding_{}.csv.gz'.format(int((epoch+1)/epochs_per_cycle)),
                        index=False,compression="gzip")
        else:
            if epoch == epochs:
                simulated_data_ls = get_encodings(model,valid_dl)
                temp= pd.DataFrame(simulated_data_ls.cpu().numpy())
                embedding.append(temp)
                if save_path:
                    pd.DataFrame(simulated_data_ls.cpu().numpy()).to_csv(
                        save_path+'_embedding_{}.csv.gz'.format(embedding_number),
                        index=False,compression="gzip")       
    return model,history,embedding
