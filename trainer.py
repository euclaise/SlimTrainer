from dataclasses import dataclass
from datasets import Dataset
from typing import Callable, Mapping
from transformers import PreTrainedModel
from tqdm import trange
from tqdm.contrib import tenumerate
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional
import wandb

from .optim import OverlapSGD

@dataclass
class SlimTrainer():
    model: PreTrainedModel
    optim: OverlapSGD
    train_data: Dataset
    epochs: int
    data_collater: Callable
    batch_size: int
    scheduler: LRScheduler
    wandb_entity: Optional[str]
    wandb_project: Optional[str]
    wandb_name: Optional[str]

    def _prepare_input(self, data):
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        if isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        if isinstance(data, torch.Tensor):
            return data.to(self.model.device)

    def train(self):
        if self.wandb_entity is not None:
            wandb.init(entity=self.wandb_entity, project=self.wandb_name, name=self.wandb_name)

        optim.init()
        loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.data_collater)

        total_batches = len(loader)

        for epoch in trange(self.epochs, desc="Epoch"):
            if hasattr(self.scheduler, "epoch_init"):
                self.scheduler.epoch_init()

            for batch_idx, batch in tenumerate(loader, desc="Batch"):
                loss = self.model(**batch).loss

                current_lr = self.scheduler.get_lr()

                self.optim.step(loss, current_lr) # Backwards pass is mixed with optimization pass

                self.scheduler.step()

                if self.wandb_entity is not None:
                    wandb.log({'loss': loss.item()})
                    wandb.log({'epoch': epoch + batch_idx / total_batches})
                    wandb.log({'step': epoch*total_batches_self.batch_size + batch_idx*self.batch_size})
                    wandb.log({'learning_rate': current_lr})

            if hasattr(self.scheduler, "epoch_end"):
                self.scheduler.epoch_end()
