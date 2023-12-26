from .trainer import SlimTrainer
from dataclasses import dataclass
import copy
import torch
from torch import nn
from typing import Dict

def generate_pairs(n):
    # Pos of i < pos of j, means that r(i) > r(j)
    for i in range(n - 1):
        for j in range(i + 1, n):
            yield (i, j)

class RankLoss:
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

class DPOLoss(RankLoss):
    def __init__(model: nn.Module, beta: float = 1.0):
        self.ref = copy.deepcopy(model)
        self.model = model
        self.beta = beta

    def pair_loss(
        self,
        logits_w_ref: torch.Tensor,
        logits_w_p: torch.Tensor,
        logits_l_ref: torch.Tensor,
        logits_l_p: torch.Tensor,
        labels_w: torch.LongTensor,
        labels_l: Torch.LongTensor
    ):


        logp_w_ref = self.logprob(logits_w_ref, lablels_w)
        logp_w_p = self.logprob(logits_w_p, lablels_w)

        logp_l_ref = self.logprob(logits_l_ref, lablels_l)
        logp_l_p = self.logprob(logits_l_p, lablels_l)

        left = self.beta * (logp_w_p - logp_w_ref)
        right = self.beta * (logp_l_p - logp_w_ref)

        return -F.logsigmoid(left - right)


    def loss(self, labels: torch.Tensor, meta: Dict, **inputs):
        # [bsz, candidates, seq_len]
        k, v = [(ks, vs) for ks, vs in inputs.items()][0]
        
        n_cands = v.shape[1]

        pair_losses = []
        pair_valid = []

        cache = dict()

        for (i, j) in generate_pairs(n_cands):
            if i in cache:
                logits_i_ref, logits_i_p = cache[i]
            else:
                logits_i_ref = self.ref(**{k: v[:, i, ...]}).logits.log_softmax(dim=-1)
                logits_i_p = self.model(**{k: v[:, i, ...]}).logits.log_softmax(dim=-1)
                cache[i] = (logits_i_ref, logits_i_p)

            if j in cache:
                logits_j_ref, logits_j_p = cache[j]
            else:
                logits_j_ref = self.ref(**{k: v[:, j, ...]}).logits.log_softmax(dim=-1)
                logits_j_p = self.model(**{k: v[:, j, ...]}).logits.log_softmax(dim=-1)
                cache[j] = (logits_j_ref, logits_j_p)

            pair_losses.append(self.pair_loss(
                logits_i_ref,
                logits_i_p,
                logits_j_ref,
                logits_j_p,
                labels[:, i, ...],
                labels[:, j, ...]
            ))
            pair_valid.append(meta['mask'][:, i] & meta['mask'][:, j])

        pair_losses = torch.stack(pair_losses) * torch.stack(pair_valid)
        return pair_losses.sum()
        
class IPOLoss(DPOLoss):
    def __init__(model: nn.Module, tau: float = 1.0):
        self.ref = copy.deepcopy(model)
        self.model = model
        self.tau = tau

    def pair_loss(
        self,
        logits_w_ref: torch.Tensor,
        logits_w_p: torch.Tensor,
        logits_l_ref: torch.Tensor,
        logits_l_p: torch.Tensor,
        labels_w: torch.LongTensor,
        labels_l: Torch.LongTensor
    ):
        logp_w_ref = self.logprob(logits_w_ref, lablels_w)
        logp_w_p = self.logprob(logits_w_p, lablels_w)

        logp_l_ref = self.logprob(logits_l_ref, lablels_l)
        logp_l_p = self.logprob(logits_l_p, lablels_l)

        h = (logp_w_p + logp_l_ref) - (logp_w_l + logp_w_ref)

        return -((h - ((1 / self.tau) / 2)) ** 2)

class cDPOLoss(DPOLoss):
    def __init__(model: nn.Module, beta: float = 1.0, eps: float = 0.05):
        self.ref = copy.deepcopy(model)
        self.model = model
        self.beta = beta
        self.eps = eps

    def pair_loss(
        self,
        logits_w_ref: torch.Tensor,
        logits_w_p: torch.Tensor,
        logits_l_ref: torch.Tensor,
        logits_l_p: torch.Tensor,
        labels_w: torch.LongTensor,
        labels_l: Torch.LongTensor
    ):
        if torch.rand() < self.eps:
            tmp = logits_w_ref, logits_w_p, labels_w
            logits_w_ref, logits_w_p, labels_w = logits_l_ref, logits_l_p, labels_l
            logits_l_ref, logits_l_p, labels_l = tmp


        logp_w_ref = self.logprob(logits_w_ref, lablels_w)
        logp_w_p = self.logprob(logits_w_p, lablels_w)

        logp_l_ref = self.logprob(logits_l_ref, lablels_l)
        logp_l_p = self.logprob(logits_l_p, lablels_l)

        left = self.beta * (logp_w_p - logp_w_ref)
        right = self.beta * (logp_l_p - logp_w_ref)

        return self.eps * F.logsigmoid(left - right) + (1 - self.eps) * F.logsigmoid(right - left)

class RRHFLoss:
    def __init__(model: nn.Module):
        self.model = model

    def pair_loss(
        self,
        logits_w: torch.Tensor,
        logits_l: torch.Tensor,
        labels_w: torch.LongTensor,
        labels_l: Torch.LongTensor
    ):
        logprob_w = self.logprob(logits_w, labels_w)
        logp_w = self.logprob(logits_w, lablels_w, normalize=True)
        logp_l = self.logprob(logits_l, lablels_l, normalize=True)

        lm_loss = -logprob_w.sum()

        ridge_loss = (logp_l - logp_w).clamp(min=0).sum()

        return lm_loss + ridge_loss


    def loss(self, labels: torch.Tensor, meta: Dict, **inputs):
        # [bsz, candidates, seq_len]
        k, v = [(ks, vs) for ks, vs in inputs.items()][0]
        
        n_cands = v.shape[1]

        pair_losses = []
        pair_valid = []

        cache = dict()

        for (i, j) in generate_pairs(n_cands):
            if i in cache:
                logits_i = cache[i]
            else:
                logits_i = self.model(**{k: v[:, i, ...]}).logits.log_softmax(dim=-1)
                cache[i] = logits_i

            if j in cache:
                logits_j = cache[j]
            else:
                logits_j = self.model(**{k: v[:, j, ...]}).logits.log_softmax(dim=-1)
                cache[j] = logits_j

            pair_losses.append(self.pair_loss(
                logits_i,
                logits_j,
                labels[:, i, ...],
                labels[:, j, ...]
            ))
            pair_valid.append(meta['mask'][:, i] & meta['mask'][:, j])

        pair_losses = torch.stack(pair_losses) * torch.stack(pair_valid)
        return pair_losses.sum()

class PROLoss:
    def __init__(model: nn.Module):
        self.model = model

    def loss(self, labels: torch.Tensor, meta: Dict, **inputs):
        # [bsz, candidates, seq_len]
        k, v = [(ks, vs) for ks, vs in inputs.items()][0]

        lm_loss = None
        logits = []
        n_cands = v.shape[1]

        for i in range(n_cands):
            output = self.model(**{k: v[:, i, ...]})

            if i == 0:
                lm_loss = output.loss

            logits.append(output.logits.log_softmax(dim=-1))

        logits = torch.stack(logits)

        cand_losses = []
        for i in range(n_cands - 1):
            num = torch.exp(self.logprob(logits[i], labels[:, i, ...], normalize=True))

            denom = torch.zeros_like(num)

            for j in range(i, n_cands - 1):
                denom += torch.exp(self.logprob(logits[j], labels[:, j, ...], normalize=True))

            r = torch.log(num / denom)
            
            r *= meta['mask'][:, i] & meta['mask'][:, j]
            cand_losses.append(r)

        r_loss = -torch.stack(cand_losses).sum()       
        return lm_loss + r_loss

    
@dataclass
class RankedTrainer(SlimTrainer):
    loss: RankLoss

    def compute_loss(self, labels: torch.Tensor, meta: Dict, **inputs):
        # [bsz, candidates, seq_len]
        k, v = [(ks, vs) for ks, vs in inputs.items()][0]
        assert meta is not None and 'mask' in meta, "Ranked modeling requires an ordering mask" 
        
        return self.loss.loss(labels, inputs, meta)
