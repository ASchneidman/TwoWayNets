import torch

class TiedLinear(torch.nn.Module):
    def __init__(self, D_in, D_out, paired_linear: TiedLinear = None):
        super(TiedLinear, self).__init__()
        self.linear = torch.nn.Linear(D_in, D_out, bias=True)
        if paired_linear is not None:
            self.linear.weight.data = paired_linear.weight.transpose(0, 1)
        
    def forward(self, x):
        return self.linear(x)