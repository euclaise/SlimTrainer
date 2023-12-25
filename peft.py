import torch
from torch.nn import functional as F
from torch import nn
from typing import List

class MoVLinear(nn.Module):
    def __init__(self, layer: nn.Linear, num_experts: int):
        super().__init__()
        self.b = nn.Parameter(layer.bias.data) if layer.bias != None else None
        self.W = nn.Parameter(layer.weight.T.data)
        self.num_steps = num_steps

        ind, outd = self.W.shape

        to = {
            'device': self.W.device,
            'dtype': self.W.dtype
        }

        self.s = nn.Linear(ind, num_experts).to(**to)
        self.v = nn.Parameter(torch.ones(num_experts, outd).to(**to))

    def forward(self, x):
        s = F.softmax(self.s(x), dim=-1)

        if self.b is not None:
            x = x @ self.W + self.b
        else:
            x = x @ self.W

        u = torch.einsum('e,ed->d', s, self.v)
        return x * u
        

def install_MoV(module: nn.Module, targets: List[str], num_experts: int):
    for name, submodule in module.named_children():
        if isinstance(submodule, nn.Linear) and name in targets:
            setattr(module, name, MoVLinear(submodule, num_experts))
            del submodule
        else:
            install_MoV(submodule, targets, num_experts)
