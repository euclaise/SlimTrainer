import torch
from torch import nn
from torch.nn import functional as F

class QuantEmbedding(nn.Module):
    def __init__(self, old_emb: nn.Embedding):
        super().__init__()

        old_weight = old_emb.weight.data

        self.embedding_dim = old_emb.embedding_dim
        self.num_embeddings = old_emb.num_embeddings

        mins = old_weight.min(dim=1).values.view(-1, 1)
        maxs = old_weight.max(dim=1).values.view(-1, 1)

        self.scales = nn.Parameter((maxs - mins) / 255.0, requires_grad=False)
        self.means = nn.Parameter(old_weight.mean(dim=1).view(-1, 1), requires_grad=False)

        self.scales[self.scales == 0] = 1e-8

        self.weight = nn.Parameter(((old_weight - self.means) / self.scales).round().clamp(min=-128, max=127).to(torch.int8), requires_grad=False)

        self.requires_grad = False

    def forward(self, idx):
        quantized_emb = self.weight[idx].float()

        return (self.scales[idx] * (quantized_emb + self.means[idx])).bfloat16()

    def dequantize(self):
        weight = (self.weight.to("cpu").float() + self.means.cpu()) * self.scales.cpu()
        res = torch.nn.Embedding(_weight = weight, embedding_dim = self.embedding_dim, num_embeddings=self.num_embeddings, dtype=torch.float32, device="cpu")
        return res.to(self.weight.device).bfloat16()

def quantembs(module):
    for name, submodule in module.named_children():
        if isinstance(submodule, nn.Embedding):
            setattr(module, name, QuantEmbedding(submodule))
            del submodule
        else:
            quantembs(submodule)

def dequantembs(module):
    for name, submodule in module.named_children():
        if isinstance(submodule, QuantEmbedding):
            setattr(module, name, submodule.dequantize())
        else:
            dequantembs(submodule)
