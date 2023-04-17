import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
import os
from math import pi
from math import cos

# cuda = True if torch.cuda.is_available() else False
# FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def snapshot_lr(initial_lr, epoch, epoch_per_cycle):
    # proposed learning late function #return initial_lr * (cos(pi * epoch / epoch_per_cycle) + 1) / 2
    return initial_lr * (cos(pi * ((epoch-1)%epoch_per_cycle) / epoch_per_cycle) + 1) / 2

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED']=str(seed)


def get_encodings(model, dl):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    result = []
    with torch.no_grad():
        for x in dl:
            encodings, mu, var = model.encoder(x.to(device))
            result.append(encodings)
    return torch.cat(result, dim=0)


def get_decodings(model, dl):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    result = []
    with torch.no_grad():
        for x in dl:
                decodings, mu, var = model(x.to(device))
                result.append(decodings)
    return torch.cat(result, dim=0)
 
    
def KL_loss(mu, logvar):
    #BCE = nn.MSELoss()(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu**2 -  logvar.exp())
    return  KLD

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    #print(mu)
    return eps*std + mu
 
from pynvml import *
import time, datetime

def nvidia_info(pid):
    # pip install nvidia-ml-py
    nvidia_dict = {}
    try:
        nvmlInit()
        #nvidia_dict["nvidia_version"] = nvmlSystemGetDriverVersion()
        nvidia_count  = nvmlDeviceGetCount()
        for i in range(nvidia_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            memory_info = nvmlDeviceGetComputeRunningProcesses(handle)
            for proc in nvmlDeviceGetComputeRunningProcesses(handle):
                if proc.pid==pid:
                    nvidia_dict = {"state": True,
                                 "pid": proc.pid,
                                 "memory": proc.usedGpuMemory/2**20,
                                 "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
                                }
    except NVMLError as _:
        nvidia_dict["state"] = False
    except Exception as _:
        nvidia_dict["state"] = False
    finally:
        try:
            nvmlShutdown()
        except:
            pass
    return nvidia_dict
