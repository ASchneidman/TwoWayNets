import torch

class TiedDropoutLayer(torch.nn.Module):
    def __init__(self, p = 0.5, paired_dropout : TiedDropoutLayer = None):
        super(TiedDropoutLayer, self).__init__()
        self.paired_dropout = paired_dropout
        self.p = p
        self.mask = None
        self.mode = True

    def eval():
        self.train(False)

    def train(mode: bool = True):
        self.mode = mode

    def forward(self, x):
        if self.mode == False or self.p == 0.0:
            return x
        retain_prob = 1 - self.p
        if self.paired_dropout is not None:
            self.mask = self.paired_dropout.mask
            assert mask is not None
        else:
            self.mask = torch.bernoulli(torch.full(x.shape, 1 - self.p))

        return (x * self.mask) / torch.sqrt(1 - self.p)