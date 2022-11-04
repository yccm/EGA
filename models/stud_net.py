import torch
import torch.nn as nn
import torch.nn.functional as F


class StudNet(nn.Module):
    def __init__(self, f_dim, feature_dim):
        # FC layers after feature extraction
        # f_dim is the feature dimension after student resnet.(depending on the number of filters of student resnet.)
        # feature_dim is the dimension of joint feature space. eg. 256 or 512.
        super(StudNet, self).__init__()
        self.fit_dim_Net = nn.Linear(f_dim, feature_dim)
        self.cl_Net = nn.Linear(feature_dim,100)        
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):        
        x = self.fit_dim_Net(x)
        fs = x
        x = self.cl_Net(x)
        logit = x        
        return fs, logit

