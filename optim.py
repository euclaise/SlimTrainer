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
        for p in self.model.parameters():
            if p.requires_grad:
                self.hook(p)

    def step(self, loss, lr):
        self.lr = lr
        loss.backward()

    def hook(self, p):
        # Gradient accumulator function for p
        # Hook this instead of p, so we know that the hook is being called post-accumulation
        acc_grad = p.view_as(p).grad_fn.next_functions[0][0]

        self._acc_grads.append(acc_grad)


        m = torch.zeros_like(p)

        def grad_func(*_):
            nonlocal m

            if p.grad is None:
                return

            with torch.no_grad():
                g = p.grad

                p.data.mul_(1 - self.lr * self.decay)

                update = m.clone().mul_(self.beta1).add_(g, alpha=1 - self.beta1).sign_()
                p.add_(update, alpha=-self.lr)

                m.mul_(self.beta2).add_(g, alpha=1 - self.beta2)

                p.grad = None
            
        acc_grad.register_hook(grad_func)


@dataclass
class OverlapSGD(OverlapOptimizer):
    def init(self):
        for p in self.model.parameters():
            if p.requires_grad:
                self.hook(p)

    def step(self, loss, lr):
        self.lr = lr
        loss.backward()

    def hook(self, p):
        # Gradient accumulator function for p
        # Hook this instead of p, so we know that the hook is being called post-accumulation
        acc_grad = p.view_as(p).grad_fn.next_functions[0][0]

        self._acc_grads.append(acc_grad)


        def grad_func(*_):
            if p.grad is None:
                return

            with torch.no_grad():
                p.add_(p.grad, alpha=-self.lr)
                p.grad = None
            
        acc_grad.register_hook(grad_func)
        #p.register_hook(grad_func)

@dataclass
class MiniLOMO(OverlapOptimizer):
    def init(self):
        for p in self.model.parameters():
            if p.requires_grad:
                p.register_hook(self.grad_func)

    def step(self, loss, lr):
        self.lr = lr
        loss.backward()

    def grad_func(x):
        for p in self.model.parameters():
            with torch.no_grad():
                if p.grad is None or not p.requires_grad:
                    return x
                p.add_(p.grad, alpha=-self.lr)
                p.grad = None
            return x
