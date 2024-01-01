from .trainer import SlimTrainer
from dataclasses import dataclass
import copy
import torch
from torch import nn
from typing import Dict

def logmeanexp(x, dim=-1):
    x_max, _ = x.max(dim=dim)
    return (x - x_max).exp().mean(dim=dim).log() + x_max

def logprob(self, logits: torch.Tensor, labels: torch.LongTensor, normalize=False):
    labels = labels.clone()

    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()

    mask = (labels != -100)
    labels[labels == -100] = 0

    logprobs = logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    if normalize:
        return (logprobs * mask).sum(dim=-1) / mask.sum(dim=-1)
    else:
        return (logprobs * mask).sum(dim=-1)

@dataclass
class PROTrainer(SlimTrainer):
    normalize_categories: bool = True

    def compute_loss(self, labels: torch.Tensor, meta: Dict, **inputs):
        # [bsz, candidates, seq_len]
        inkk, inv = [(ks, vs) for ks, vs in inputs.items()][0] # input_ids/input_embeds

        lm_loss = None
        logits = []
        max_n_cands = inv.shape[1]

        for i in range(n_cands):
            output = self.model(**{k: v[:, i, ...]})

            if i == 0:
                lm_loss = output.loss

            logits.append(output.logits.log_softmax(dim=-1))

        logits = torch.stack(logits, dim=1)
        lps = [logprob(logits[:, i], labels[:, i]) for i in range(n_cands, normalize=True)]
        lps = torch.stack(lps, dim=1)

        cand_losses = []
        for i in range(lps.shape[1] - 1):
            lognum = lps[:, i]

            den_lps = lps[:, i:]
            den_lps = torch.where(meta['mask'][:, i:], den_lps, float('-inf'))

            if self.normalize_categories:
                logdenom = logmeanexp(den_lps, dim=1)
            else:
                logdenom = torch.logsumexp(den_lps, dim=1)

            r = (lognum - logdenom) * meta['mask'][:, i]
            cand_losses.append(r)

        r_loss = -torch.stack(cand_losses).sum()       
        return lm_loss + r_loss
