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
class OverlapLion(OverlapOptimizer):
    beta1: float = 0.9
    beta2: float = 0.99

    def prepare(self, p):
        return

    def step(self, loss, lr):
        self.lr = lr
        loss.backward()

    def prepare(self, p):
        p._m = torch.zeros_like(p)

    def hook(self, p):
        ag = p.view_as(p).grad_fn.next_functions[0][0]
        p._acc_grads = [ag]

        @torch.no_grad()
        def gf(*_):
            p.data.mul_(1 - self.lr * self.decay)

            update = p._m.clone().mul_(self.beta1).add_(p.grad, alpha=1 - self.beta1).sign_()
            p.add_(update, alpha=-self.lr)

            p._m.mul_(self.beta2).add_(g, alpha=1 - self.beta2)

            p.grad = None

        ag.register_hook(gf)

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
    eps: float = 1e-6
    beta_decay: float = 0.8
    centralize: bool = True
    _t: int = 0

    def step(self, loss, lr=None):
        self._t += 1
        self.lr = lr
        loss.backward()

    def prepare(self, p):
        if len(p.shape) == 2:
            n = p.shape[0]
            m = p.shape[1]

            p._c = torch.zeros(p.shape[1], device=p.device, dtype=p.dtype)

        else:
            p._v = torch.zeros_like(p)

    def hook(self, p):
        ag = p.view_as(p).grad_fn.next_functions[0][0]
        p._acc_grads = [ag]

        @torch.no_grad()
        def gf(*_):
            alpha = self.lr

            g = p.grad

            if self.centralize and sum(g.shape) > 1:
                g.add_(g.mean(dim=tuple(range(1, len(g.shape))), keepdim=True))

            beta_t = 1.0 - math.pow(self._t, -self.beta_decay)
            u = g.square() + self.eps

            if len(p.shape) == 2:
                c = u.mean(dim=0)
                p._c.addcmul_((p._c - c).sign(), c, alpha=-(1-beta_t))

                m = p._c.rsqrt().broadcast_to(g.shape)
                m = m * g
            else:
                p._v.addcmul_((p._c - u).sign(), u, alpha=-(1-beta_t))
                m = p._v.rsqrt() * g

            p_norm = p.norm()
            g_norm = g.norm()

            if p_norm != 0 and g_norm != 0:
                m.mul_(p_norm / g_norm)

            p.add_(m, alpha=-alpha)
            p.grad = None

        ag.register_hook(gf)
