import torch
from dataclasses import dataclass, field
from typing import List, Optional
from collections import defaultdict

# Somewhat based on https://gist.github.com/albanD/18c240bd2e09f9d93f5c4a0c9ccda39e and LOMO
@dataclass
class OverlapLion(torch.optim.Optimizer):
    model: torch.nn.Module
    lr: Optional[float] = None
    decay: Optional[float] = 0.0
    defaults: List = field(default_factory=lambda: [])
    beta1: float = 0.9
    beta2: float = 0.99

    _acc_grads: Optional[List] = field(default_factory=lambda: [])

    def init(self):
        if self.pastnorm:
            self._norm_sum = torch.tensor(0.0, requires_grad=False, device=self.model.device)

        for p in self.model.parameters():
            p._exp_avg = torch.tensor(0, requires_grad=False, device=self.model.device, dtype=torch.int8)
            if p.requires_grad:
                self.hook(p)

    def step(self, loss, lr):
        self.lr = lr
        loss.backward()

    def grad_fn(self, g):
        if self.sign:
            g = g.sign()

        return g

    def hook(self, p):
        # Gradient accumulator function for p
        # Hook this instead of p, so we know that the hook is being called post-accumulation
        acc_grad = p.view_as(p).grad_fn.next_functions[0][0]

        self._acc_grads.append(acc_grad)

        def grad_func(*_):
            with torch.no_grad():
                p.data.mul_(1 - self.lr * self.decay)

                update = p._exp_avg * self.beta1 + g * (1 - beta1)

                p.data.add_(-self.lr * torch.sign(update), inplace=True)
                p._exp_avg.data.mul(beta2).add_(grad, alpha=1 - beta2)

                p.grad = None
            
        acc_grad.register_hook(grad_func)
