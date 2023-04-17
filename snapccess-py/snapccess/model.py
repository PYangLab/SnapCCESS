import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class LinBnDrop(nn.Sequential):
    """Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers, adapted from fastai."""
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=True):
        layers = [nn.BatchNorm1d(n_out if lin_first else n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)

## nn.module is the superclass, encoder is the subclass
class Encoder(nn.Module):
    """Encoder for multi-modal data"""
    def __init__(self, num_features: list, num_hidden_features: list, z_dim: int=128):
        super().__init__()
        self.features=num_features
        self.encoder_eachmodal= nn.ModuleList([LinBnDrop(num_features[i], num_hidden_features[i], p=0.2, act=nn.ReLU())
                                 for i in range(len(num_hidden_features))]).to(device) 
        self.encoder = LinBnDrop(sum(num_hidden_features), z_dim, act=nn.ReLU()).to(device)
        self.weights=[]
        for i in range(len(num_features)):
            self.weights.append(nn.Parameter(torch.rand(1,num_features[i]) * 0.001, requires_grad=True).to(device))# 
        self.fc_mu =nn.Sequential( LinBnDrop(z_dim,z_dim, p=0.1),#0.1
                                 ).to(device)
        self.fc_var =nn.Sequential( LinBnDrop(z_dim,z_dim, p=0.1),#0.1
                                     ).to(device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        X = []
        startfeature=0
        for i, eachmodal in enumerate(self.encoder_eachmodal):
            tmp=eachmodal(x[:,startfeature:(startfeature+self.features[i])]*self.weights[i])
            startfeature=startfeature+self.features[i]
            X.append(tmp)
        x = torch.cat(X, 1)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        x = self.reparameterize(mu, var)
        return x,mu,var
    
    
    
class Decoder(nn.Module):
    """Decoder for 2 modal data"""
    def __init__(self, num_features: list, z_dim: int = 128):
        super().__init__()
        self.features=num_features
        self.decoder_eachmodal= nn.ModuleList([ LinBnDrop(z_dim, num_features[i], act=nn.ReLU()) for i in range(len(num_features))]).to(device) 

    def forward(self, x):
        X = []
        for i, deachmodal in enumerate(self.decoder_eachmodal):
            tmp=deachmodal(x)
            X.append(tmp)
        x = torch.cat(X, 1)
        return x

class snapshotVAE(nn.Module):
    def __init__(self, num_features: list, num_hidden_features: list, z_dim: int = 20):
        super().__init__()
        self.encoder = Encoder(num_features, num_hidden_features, z_dim)
        self.decoder = Decoder(num_features, z_dim)
    def forward(self, x):
        x,mu,var = self.encoder(x)
        x = self.decoder(x)
        return x,mu,var
    
