from dataclasses import dataclass
import math
from datasets import Dataset
from typing import Callable, Mapping
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from tqdm import trange
from tqdm.contrib import tenumerate
from torch.utils.data import DataLoader
import torch
from torch import nn
from typing import Optional
import wandb

from .optim import OverlapOptimizer
from .lr import LRScheduler

def prepare_for_ranked(tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel):
    tokenizer.add_special_tokens(["[REF]"])
    model.resize_token_embeddings(len(tokenizer))
    return (tokenizer, model)

def dpo_loss(pi_logps, ref_logps, beta):
    pi_yw_logps,  pi_yl_logps =  pi_logps[:, 0],  pi_logps[:, 1:]
    ref_yw_logps, ref_yl_logps = ref_logps[:, 0], ref_logps[:, 1:]

    pi_logratios  = pi_yw_logps - pi_yl_logps
    ref_logratios = ref_yw_logps - ref_yl_logps

    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    rewards = beta * (pi_logps - ref_logps).detach()

    return losses, rewards

@dataclass
class SlimRankTrainer:
    model: PreTrainedModel
    optim: OverlapOptimizer
    ref_token: string = "[REF]"
    train_data: Dataset
    epochs: int
    data_collator: Callable
    batch_size: int
    scheduler: LRScheduler
    wandb_entity: Optional[str]
    wandb_project: Optional[str]
    wandb_name: Optional[str]
    report_steps: int = 8
    neft: bool = False

    def train(self):
        first = True
        if self.wandb_entity is not None:
            wandb.init(entity=self.wandb_entity, project=self.wandb_project, name=self.wandb_name)

        loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.data_collator)

        total_batches = len(loader)
        self.optim.init()

        ref_id = tokenizer.convert_tokens_to_ids(self.ref_token)

        if self.neft:
            embedding_layer = self.model.get_input_embeddings()

        for epoch in trange(self.epochs, desc="Epoch"):
            if hasattr(self.scheduler, "epoch_init"):
                self.scheduler.epoch_init()

            loss_avg = 0
            for batch_idx, batch in tenumerate(loader, desc="Batch"):
                if self.neft:
                    embeds = embedding_layer(batch['input_ids'].cuda())
                    noise = (torch.rand_like(embeds) - 0.5) * 10/math.sqrt(512 * embeds.shape[-1])
                    losses = self.model(
                        inputs_embeds=torch.where(batch['input_ids'] != ref_id, embeds + noise, embeds),
                        labels=batch['labels'].cuda()
                    ).loss
                else:
                    losses = self.model(
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
