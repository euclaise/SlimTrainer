from dataclasses import dataclass
import math
from datasets import Dataset
from typing import Callable, Mapping
from transformers import PreTrainedModel
from tqdm import trange
from tqdm.contrib import tenumerate
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from typing import Optional
import wandb

from .optim import OverlapOptimizer
from .lr import LRScheduler

@dataclass
class SlimTrainer():
    model: PreTrainedModel
    optim: OverlapOptimizer
    scheduler: LRScheduler
    train_data: Dataset
    epochs: int
    data_collator: Callable
    batch_size: int
    wandb_entity: Optional[str]
    wandb_project: Optional[str]
    wandb_name: Optional[str]
    report_steps: int = 8
    neft: bool = False # https://arxiv.org/abs/2310.05914
    neft_alpha: int = 5
    freeze_embeds: bool = True
    mixce: bool = False # https://arxiv.org/abs/2305.16958
    mixce_ratio: float = 0.5
    encdec: bool = False # Encoder-decoder model

    def compute_loss(self, labels, meta, **inputs):
        if self.encdec:
            return self.model(labels=labels, **inputs).loss

        outputs = self.model(labels=labels, **inputs)
        if self.mixce:
            labels = labels.clone()

            logits = outputs.logits.log_softmax(-1)
            
            logits = logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()

            mask = (labels != -100)
            labels[labels == -100] = 0

            log_probs = logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                q = torch.exp(log_probs.detach())

            losses = self.mixce_ratio * -log_probs + (1.0 - self.mixce_ratio) * q * -log_probs
            return (losses * mask.float()).sum() / mask.sum()
        else:
            return outputs.loss

    def train(self):
        first = True
        if self.wandb_entity is not None:
            wandb.init(entity=self.wandb_entity, project=self.wandb_project, name=self.wandb_name)

        loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.data_collator)

        total_batches = len(loader)

        if self.freeze_embeds and not self.encdec:
            self.model.get_input_embeddings().requires_grad = False
            self.model.get_output_embeddings().requires_grad = False

        self.optim.init()

        if self.neft:
            embedding_layer = self.model.get_input_embeddings()

        step = 0
        for epoch in trange(self.epochs, desc="Epoch"):
            if hasattr(self.scheduler, "epoch_init"):
                self.scheduler.epoch_init()

            loss_avg = 0
            for batch_idx, batch in tenumerate(loader, desc="Batch"):
                if self.neft:
                    assert not self.encdec
                    embeds = embedding_layer(batch['input_ids'].to(self.model.device))
                    noise = (torch.rand_like(embeds) - 0.5) * self.neft_alpha/math.sqrt(512 * embeds.shape[-1])
                    loss = self.compute_loss(
                        labels=batch['labels'].to(self.model.device),
                        inputs_embeds=embeds + noise,
                        meta=batch.get('meta', None)
                    )
                else:
                    loss = self.compute_loss(
                        labels=batch['labels'].to(self.model.device),
                        input_ids=batch['input_ids'].to(self.model.device)
                        meta=batch.get('meta', None)
                    )

                self.optim.step(loss, self.scheduler.get_lr()) # Backwards pass is mixed with optimization pass
                self.scheduler.step()
                loss_avg += loss.detach().item()


                if (batch_idx + 1) % self.report_steps == 0:
                    if self.wandb_entity is not None:
                        wandb.log({'loss': loss_avg / self.report_steps}, step=step)
                        wandb.log({'epoch': epoch + batch_idx / total_batches}, step=step)
                        wandb.log({'learning_rate': self.scheduler.get_lr()}, step=step)
                    loss_avg = 0

                step += 1

            if hasattr(self.scheduler, "epoch_end"):
                self.scheduler.epoch_end()
