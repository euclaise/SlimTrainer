import math

class LRScheduler():
    def __init__(self, optimizer, total_steps, warmup_steps, max_lr=1e-3):
        pass
    def step(self):
        self.steps += 1

    def get_lr(self):
        pass

class CosineDecayWithWarmup(LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_steps, max_lr=1e-3):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.steps = 1

    def step(self):
        self.steps += 1

    def get_lr(self):
        if self.steps < self.warmup_steps:
            lr = self.max_lr * (self.steps / self.warmup_steps)
        else:
            remaining_steps = self.steps - self.warmup_steps
            total_cosine_steps = self.total_steps - self.warmup_steps
            cos_inner = (math.pi * remaining_steps) / total_cosine_steps
            lr = self.max_lr * (1 + math.cos(cos_inner)) / 2

        return lr
