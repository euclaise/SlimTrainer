from dataclasses import dataclass
from datasets import Dataset
from typing import Callable, Mapping
from transformers import PreTrainedModel
from tqdm import trange
from tqdm.contrib import tenumerate
from torch.utils.data import DataLoader
import torch
from typing import Optional
import wandb

from .optim import OverlapOptimizer
from .lr import LRScheduler

@dataclass
class SlimTrainer():
    model: PreTrainedModel
    optim: OverlapOptimizer
    train_data: Dataset
    epochs: int
    data_collater: Callable
    batch_size: int
    scheduler: LRScheduler
    wandb_entity: Optional[str]
    wandb_project: Optional[str]
    wandb_name: Optional[str]
    report_steps: int = 8
    neft: bool = False
    freeze_embeds: bool = True

    def train(self):
        first = True
        if self.wandb_entity is not None:
            wandb.init(entity=self.wandb_entity, project=self.wandb_project, name=self.wandb_name)

        loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.data_collater)

        total_batches = len(loader)
        self.optim.init()

        if self.freeze_embeds:
            model.get_input_embeddings().requires_grad = False
            model.get_output_embeddings().requires_grad = False

        if self.neft:
            embedding_layer = model.get_input_embeddings()

        for epoch in trange(self.epochs, desc="Epoch"):
            if hasattr(self.scheduler, "epoch_init"):
                self.scheduler.epoch_init()

            loss_avg = 0
            for batch_idx, batch in tenumerate(loader, desc="Batch"):
                if self.neft:
                    embeds = embedding_layer(batch['input_ids'].cuda())
                    noise = torch.rand_like(embeds) * 5/torch.sqrt(2048 * embeds.shape[-1])
                    loss = self.model(
                        input_embeds=embeds + noise,
                        labels=batch['labels'].cuda()
                    ).loss
                else:
                    loss = self.model(
                        input_ids=batch['input_ids'].cuda(),
                        labels=batch['labels'].cuda()
                    ).loss

                self.optim.step(loss, self.scheduler.get_lr()) # Backwards pass is mixed with optimization pass
                self.scheduler.step()
                loss_avg += loss.detach().item()


                if (batch_idx + 1) % self.report_steps == 0:
                    if self.wandb_entity is not None:
                        wandb.log({'loss': loss_avg / self.report_steps})
                        wandb.log({'epoch': epoch + batch_idx / total_batches})
                        wandb.log({'step': epoch*len(loader) + batch_idx})
                        wandb.log({'learning_rate': self.scheduler.get_lr()})
                    loss_avg = 0

            if hasattr(self.scheduler, "epoch_end"):
                self.scheduler.epoch_end()
