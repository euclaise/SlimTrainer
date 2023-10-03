import torch
from torch.nn import functional as F
from torch import nn

class SigmaLinear(nn.Module):
    def __init__(self, layer: nn.Linear, num_steps=1, num_steps_start=5):
        super().__init__()
        self.b = nn.Parameter(layer.bias.data) if layer.bias != None else None
        self.W = nn.Parameter(layer.weight.T.data)
        self.num_steps = num_steps

        d, c = self.W.shape

        to = {
            'device': self.W.device,
            'dtype': self.W.dtype
        }
        
        self.u = F.normalize(torch.normal(0, 1, size=(d,)), dim=0).to(**to)
        self.v = F.normalize(torch.normal(0, 1, size=(c,)), dim=0).to(**to)

        self.gamma = nn.Parameter(self._get_sigma(num_steps_start))

    def _get_sigma(self, steps):
        with torch.no_grad():
            for i in range(steps):
                self.u = F.normalize(self.W @ self.v, dim=0)
                self.v = F.normalize(self.W.T @ self.u, dim=0)
        return torch.einsum('d,dc,c->', self.u, self.W, self.v)

    def to_linear(self):
        l = nn.Linear(self.W.shape[1], self.W.shape[0], bias=(self.b is not None), device=self.W.device, dtype=self.W.dtype)
        l.weight.data = self.gamma / self._get_sigma(self.num_steps) * self.W.T
        if self.b is not None:
            l.bias.data = self.b

        return l

    def forward(self, x):
        x = x * self.gamma / self._get_sigma(self.num_steps)
        if self.b is not None:
            return x @ self.W + self.b
        else:
            return x @ self.W

def sigmafy(module):
    for name, submodule in module.named_children():
        if isinstance(submodule, nn.Linear) and "embed" not in name and "lm_head" not in name:
            setattr(module, name, SigmaLinear(submodule))
            del submodule
        else:
            sigmafy(submodule)

def desigmafy(module):
    for name, submodule in module.named_children():
        if isinstance(submodule, SigmaLinear):
            setattr(module, name, submodule.to_linear())
        else:
            desigmafy(submodule)

