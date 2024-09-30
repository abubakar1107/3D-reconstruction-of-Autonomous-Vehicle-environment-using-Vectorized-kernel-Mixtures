import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm

def to_device(data, device):
    """
    Move tensor(s) to the specified device
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def get_device():
    """
    Get the available device (CPU or GPU)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def strict_standard_normal(d):
    # this function generate very similar outcomes as torch.randn(d)
    # but the numbers are strictly standard normal, no randomness.
    y = np.linspace(0, 1, d+2)
    x = norm.ppf(y)[1:-1]
    np.random.shuffle(x)
    x = torch.tensor(x).float()
    return x

class ComplexReLU(nn.Module):
     def forward(self, real, imag):
         return F.relu(real), F.relu(imag)

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, real, imag):
        return self.fc_r(real) - self.fc_i(imag), self.fc_r(imag) + self.fc_i(real) 

class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ComplexConv1d, self).__init__()
        self.conv_r = nn.Conv1d(in_channels, out_channels, 1)
        self.conv_i = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, real, imag):
        return self.conv_r(real) - self.conv_i(imag), self.conv_r(imag) + self.conv_i(real)

class NaiveComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, \
                 track_running_stats=True):
        super(NaiveComplexBatchNorm1d, self).__init__()
        self.bn_r = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, real, imag):
        return self.bn_r(real), self.bn_i(imag)

class NormalEstimator(nn.Module):
    def __init__(self, d=256, alpha_list=[60], beta_list=[10]):
        super().__init__()
        self.device = get_device()

        self.alpha_list = alpha_list
        self.beta_list = beta_list
        self.n_reso = len(alpha_list)
        self.n_scales = len(self.beta_list)
        self.sqrt_d = d ** 0.5
        print(f'alpha_list: {self.alpha_list}; beta_list: {self.beta_list}; d: {d}')

        self.T = []
        for alpha in self.alpha_list:
            T = torch.stack(
                [strict_standard_normal(d) for _ in range(3)], 
                dim=0
            ) * alpha
            T = to_device(T, self.device)  # Move T to device after creation
            self.T.append(T)
        self.T = torch.stack(self.T, dim=0)
        self.T = nn.Parameter(self.T, False)                                    # (n_reso, 3, d)
        # self.T = to_device(self.T, self.device)

        self.W = []
        for beta in self.beta_list:
            W = torch.stack(
                [strict_standard_normal(2048) for _ in range(3)], 
                dim=0
            ) * beta
            W = to_device(W, self.device)
            self.W.append(W)
        self.W = torch.stack(self.W, dim=0)
        self.W = nn.Parameter(self.W, False)                                    # (n_scales, 3, d)
        # self.W = to_device(self.W, self.device)

        self.feat_trans1 = to_device(ComplexConv1d(256, 128), self.device)
        self.feat_trans2 = to_device(ComplexConv1d(128, 64), self.device)
        self.feat_bn1 = to_device(NaiveComplexBatchNorm1d(128), self.device)
        self.feat_bn2 = to_device(NaiveComplexBatchNorm1d(64), self.device)

        self.out_fc = to_device(nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 3)
        ), self.device)

        self.relu = ComplexReLU()

    def forward(self, pts, normal):
        pts = to_device(pts.unsqueeze(dim=0), self.device)                                             # (1, n, 3)
        # T_cuda = self.T.cuda()  # Create a new tensor on the GPU
        # W_cuda = self.W.cuda()  # Create a new tensor on the GPU
        eT = torch.exp(1j * (pts @ self.T)).unsqueeze(0)  # (1, n_reso, n, d)
        eW = torch.exp(1j * (pts @ self.W)).unsqueeze(1)  # (n_scales, 1, n, d)

        G = torch.matmul(
            eW, 
            eW.transpose(-1,-2).conj() @ eT
        ) / eT                                                                  # (n_scales, n_reso, n, d)
        G = G / torch.norm(G, dim=-1, keepdim=True) * self.sqrt_d               # (n_scales, n_reso, n, d)
        G = G.reshape(-1, G.shape[-2], G.shape[-1])                             # (n_scales * n_reso, n, d)
        G = G.permute(1,2,0) 
        
        # print(G.shape)                                                 # (n, d, n_scales * n_reso)

        real, imag = self.feat_trans1(G.real, G.imag)                           # (n, 512, n_scales * n_reso)
        # print(real.shape)
        # dfdf
        real, imag = self.feat_bn1(real, imag)                                  # (n, 512, n_scales * n_reso)
        real, imag = self.relu(real, imag)                                      # (n, 512, n_scales * n_reso)
        real, imag = self.feat_trans2(real, imag)                               # (n, 256, n_scales * n_reso)
        real, imag = self.feat_bn2(real, imag)                                  # (n, 256, n_scales * n_reso)

        G = real**2 + imag**2
        G = torch.max(G, dim=-1)[0]                                             # (n, 256)

        pred = self.out_fc(G)
        return pred, normal

