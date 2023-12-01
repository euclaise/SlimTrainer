import torch
import math
from dataclasses import dataclass, field
from typing import List, Optional
from collections import defaultdict

# Somewhat based on https://gist.github.com/albanD/18c240bd2e09f9d93f5c4a0c9ccda39e and LOMO

@dataclass
class OverlapOptimizer:
    model: torch.nn.Module
    lr: Optional[float] = None
    decay: Optional[float] = 0.0
    _acc_grads: Optional[List] = field(default_factory=lambda: [])

    def init(self):
        for p in self.model.parameters():
            if p.requires_grad:
                self.prepare(p)
                self.hook(p)

    def step(self, loss, lr):
        pass

    def hook(self, p):
        pass

@dataclass
class OverlapSGD(OverlapOptimizer):
    sign: bool = False

    def prepare(self, p):
        return

    def step(self, loss, lr):
        self.lr = lr
        loss.backward()

    def hook(self, p):
        ag = p.view_as(p).grad_fn.next_functions[0][0]
        p._acc_grads = [ag]

        @torch.no_grad()
        def gf(*_):
            if self.sign:
                p.add_(p.grad.sign(), alpha=-self.lr)
            else:
                p.add_(p.grad, alpha=-self.lr)
            p.grad = None

        ag.register_hook(gf)

@dataclass
class Adalite(OverlapOptimizer):
    eps: float = 1e-5
    Lambda: float = 0.01
    beta_decay: float = 0.8
    centralize: bool = True
    momentum: bool = False
    momentum_beta: float = 0.9
    _t: int = 0

    def step(self, loss, lr=None):
        self._t += 1
        self.lr = lr
        loss.backward()

    def prepare(self, p):
        if len(p.shape) == 2:
            p._c = torch.zeros(p.shape[1], device=p.device, dtype=p.dtype)
        else:
            p._v = torch.zeros_like(p)

        if self.momentum:
            p._m = torch.zeros_like(p)

    def hook(self, p):
        ag = p.view_as(p).grad_fn.next_functions[0][0]
        p._acc_grads = [ag]

        @torch.no_grad()
        def gf(*_):
            alpha = self.lr

            g = p.grad

            if self.centralize and sum(g.shape) > 1:
                g.sub_(g.mean(dim=tuple(range(1, len(g.shape))), keepdim=True))

            beta_t = 1.0 - math.pow(self._t, -self.beta_decay)
            u = g.square()

            if len(p.shape) == 2:
                u.mul_(1-beta_t).add_(p._c.unsqueeze(0).broadcast_to(g.shape), alpha=beta_t)
                u.add_(self.eps)
                p._c = u.mean(dim=0)
            else:
                u.mul_(1-beta_t).add_(p._v, alpha=beta_t)
                u.add_(self.eps)
                p._v = u

            m = u.rsqrt() * g

            p_norm = p.norm()
            g_norm = g.norm()

            if p_norm != 0 and g_norm != 0:
                m.mul_(p_norm / g_norm)
                m.add_(p - p/p_norm, alpha=self.Lambda)

            if self.momentum:
                p._m.mul_(self.momentum_beta).add_(m, alpha=1-self.momentum_beta)
                m = p._m

            p.add_(m, alpha=-alpha)
            p.grad = None

        ag.register_hook(gf)
