import numpy
import torch

class RegularizedBatchNorm(torch.nn.Module):
    def __init__(self, D_in, 
                       gamma=numpy.random.uniform([.95, 1.05]), 
                       beta=0.0, 
                       epsilon=0.001):
        super(RegularizedBatchNorm, self).__init__()
        self.gamma = torch.nn.Parameter(torch.full((1, D_in), gamma[0]))
        self.gamma.requires_grad = True
        self.beta = beta
        self.epsilon = epsilon
        self.D_in = D_in
    
    def forward(self, x):
        m = x.mean(axis=0, keepdim=True)
        v = torch.sqrt(torch.var(x, axis=0, keepdim=True) + self.epsilon)
        x_hat = (x - m) / v
        return x_hat * self.gamma + self.beta