import torch
from dataclasses import dataclass, field
from typing import List, Optional

# Somewhat based on https://gist.github.com/albanD/18c240bd2e09f9d93f5c4a0c9ccda39e and LOMO
@dataclass
class OverlapSGD(torch.optim.Optimizer):
    model: torch.nn.Module
    lr: Optional[float] = None
    sign: bool = False
    decay: Optional[float] = 0.0
    pastnorm: bool = False
    norm_smooth: float = 0.3
    norm_clip: Optional[float] = None
    param_groups: List = field(default_factory=lambda: [{'lr': 0.01}])
    defaults: List = field(default_factory=lambda: [])

    _norm: Optional[torch.Tensor] = None
    _norm_sum: Optional[torch.Tensor] = None
    _acc_grads: Optional[List] = field(default_factory=lambda: [])

    def init(self):
        if self.pastnorm:
            self._norm_sum = torch.tensor(0.0, requires_grad=False, device=self.model.device)

        for p in self.model.parameters():
            if p.requires_grad:
                self.hook(p)

    def step(self, loss, lr):
        if self.pastnorm:
            if self._norm == None or self._norm_sum == 0.0:
                self._norm = torch.sqrt(self._norm_sum)
            else:
                self._norm = (1 - self.norm_smooth)*self._norm + self.norm_smooth*torch.sqrt(self._norm_sum)
            self._norm_sum = torch.tensor(0.0, requires_grad=False, device=self.model.device)
        self.lr = lr
        loss.backward()

    def grad_fn(self, g):
        norm = self._norm

        if self.sign:
            g = g.sign()

        if self.pastnorm:
            self._norm_sum += g.norm(2) ** 2
            if norm != None:
                g /= norm

        if self.norm_clip is not None:
            clip_val = self.norm_clip*norm
            g = torch.clamp(g, min=-clip_val, max=clip_val)

        return g

    def hook(self, p):
        # Gradient accumulator function for p
        # Hook this instead of p, so we know that the hook is being called post-accumulation
        acc_grad = p.view_as(p).grad_fn.next_functions[0][0]

        self._acc_grads.append(acc_grad)

        def grad_func(*_):
            with torch.no_grad():
                g = self.grad_fn(p.grad)

                if self.decay is not None:
                    g += self.decay * p

                p.data.add_(-self.lr * g)
                p.grad = None
            
        acc_grad.register_hook(grad_func)

    def set_lr(self, new_lr):
        self.lr = new_lr
        for param_group in self.param_groups:
            param_group['lr'] = new_lr
