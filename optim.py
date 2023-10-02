import torch
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
        pass

    def step(self, loss, lr):
        pass

    def hook(self, p):
        pass

@dataclass
class OverlapLion(OverlapOptimizer):
    beta1: float = 0.9
    beta2: float = 0.99

    def init(self):
        grad_func = self.grad_func()

        for p in self.model.parameters():
            if p.requires_grad:
                self.prepare(p)
                p.register_hook(grad_func)

    def step(self, loss, lr):
        self.lr = lr
        loss.backward()

    def prepare(self, p):
        p._m = torch.zeros_like(p)

    def grad_func(self):
        @torch.no_grad()
        def gf(x):
            for p in self.model.parameters():
                if not p.requires_grad or p.grad is None:
                    continue

                g = p.grad

                p.data.mul_(1 - self.lr * self.decay)

                update = p._m.clone().mul_(self.beta1).add_(g, alpha=1 - self.beta1).sign_()
                p.add_(update, alpha=-self.lr)

                p._m.mul_(self.beta2).add_(g, alpha=1 - self.beta2)

                p.grad = None
            return x
        return gf

@dataclass
class OverlapSGD(OverlapOptimizer):
    sign: bool = False
    def init(self):
        grad_func = self.grad_func()
        for p in self.model.parameters():
            if p.requires_grad:
                p.register_hook(grad_func)

    def step(self, loss, lr):
        self.lr = lr
        loss.backward()

    def grad_func(self):
        @torch.no_grad()
        def gf(x):
            for p in self.model.parameters():
                if not p.requires_grad or p.grad is None:
                    continue
                if self.sign:
                    p.add_(p.grad.sign(), alpha=-self.lr)
                else:
                    p.add_(p.grad, alpha=-self.lr)
                p.grad = None
            return x
        return gf

@dataclass
class Serval(OverlapOptimizer):
    def init(self):
        grad_func = self.grad_func()

        for p in self.model.parameters():
            if p.requires_grad:
                self.prepare(p)
                p.register_hook(grad_func)

    def step(self, loss, lr):
        self.lr = lr
        loss.backward()

    def prepare(self, p):
        p._n = torch.zeros_like(p.norm(ord=2))

    def grad_func(self):
        @torch.no_grad()
        def gf(x):
            for p in self.model.parameters():
                if not p.requires_grad or p.grad is None:
                    continue

                g = p.grad
                gn = g.norm(ord=2)

                p.data.mul_(1 - self.lr * self.decay)

                update = p._n.clone().mul_(self.beta1) + (gn * (1 - self.beta1))).sign_()
                p.add_(update, alpha=-self.lr)

                p._n.mul_(self.beta2).add_(gn, alpha=1 - self.beta2)

                p.grad = None
            return x
        return gf
