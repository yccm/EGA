import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacNet(nn.Module):

    def __init__(self, f_dim, feature_dim):
        # FC layers after feature extraction
        # f_dim is the feature dimension after clip.(depending on which pre-trained clip model is used.) eg.512, 1024.
        # feature_dim is the dimension of joint feature space. eg. 256 or 512.
        super(TeacNet, self).__init__()
        self.fit_dim_Net = nn.Linear(f_dim, feature_dim)
        self.cl_Net = nn.Linear(feature_dim,100) 
    def forward(self, x):
        x = self.fit_dim_Net(x)
        ft = x
        x = self.cl_Net(x)
        logit = x        
        return ft, logit

