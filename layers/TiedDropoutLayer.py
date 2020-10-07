import torch, math

class TiedDropoutLayer(torch.nn.Module):
    def __init__(self, p = 0.5, paired_dropout = None):
        super(TiedDropoutLayer, self).__init__()
        self.paired_dropout = paired_dropout
        self.p = p
        self.mask = None
        self.mode = True

    def eval(self):
        self.train(False)

    def train(self, mode: bool = True):
        self.mode = mode

    def forward(self, x):
        if self.mode == False or self.p == 0.0:
            self.mask = torch.full(x.shape, 1.0)
            return x
        retain_prob = 1 - self.p
        if self.paired_dropout is not None:
            self.mask = self.paired_dropout.mask
            assert self.mask is not None
        else:
            self.mask = torch.bernoulli(torch.full(x.shape, 1 - self.p))

        self.mask = self.mask.to(x.device)
        return (x * self.mask) / math.sqrt(1 - self.p)