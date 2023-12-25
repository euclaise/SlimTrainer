import torch
from torch.nn import functional as F
from torch import nn
from typing import List, Dict
from transformers.activations import ACT2FN

class MoVLinear(nn.Module):
    def __init__(self, layer: nn.Linear, num_experts: int):
        super().__init__()
        self.linear = layer

        to = {
            'device': self.linear.weight.device,
            'dtype': self.linear.weight.dtype
        }

        self.s = nn.Linear(self.linear.weight.shape[1], num_experts).to(**to)
        self.v = nn.Parameter(torch.ones(num_experts, self.linear.shape[0]).to(**to))

    def forward(self, x):
        s = F.softmax(self.s(x), dim=-1)
        u = torch.einsum('e,ed->d', s, self.v)
            
        x = self.linear(x)

        return x * u


class MoVGLU(nn.Module):
    def __init__(self, mlp: nn.Module, config: Dict, num_experts: int):
        super().__init__()
        self.config = config

        if config['gate'] != None:
            self.gate = getattr(mlp, config['gate'])

        self.up = getattr(mlp, config['up'])
        self.down = getattr(mlp, config['down'])
        self.act = getattr(mlp, config['act'])

        to = {
            'device': self.up.weight.device,
            'dtype': self.up.weight.dtype
        }

        self.s = nn.Linear(self.up.weight.shape[1], num_experts).to(**to)
        self.v = nn.Parameter(torch.ones(num_experts, self.up.shape[0]).to(**to))

    def forward(self, x):
        s = F.softmax(self.s(x), dim=-1)
        u = torch.einsum('e,ed->d', s, self.v)

        if self.config['gate'] != None:
            x = self.down(self.act(self.gate(x)) * self.up(x) * u)
        else:
            x = self.down(self.act(self.up(x)) * u)

        return x
        

def install_MoV(
        module: nn.Module,
        linear_targets: List[str],
        mlp_targets: List[str],
        num_experts: int,
        mlp_config: Dict = {
            'up': 'up_proj',
            'gate': 'gate_proj',
            'down': 'down_proj',
            'act': 'act_fn'
        },
    ):
    for name, submodule in module.named_children():
        if isinstance(submodule, nn.Linear) and name in linear_targets:
            setattr(module, name, MoVLinear(submodule, num_experts))
            del submodule
        elif name in mlp_targets:
            setattr(module, name, MoVGLU(submodule, mlp_config, num_experts))
            del submodule
        else:
            install_MoV(submodule, targets, num_experts)
