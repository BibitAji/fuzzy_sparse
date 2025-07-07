import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# ===== FuzzySigmoidLayer with OWA Ranking =====
class FuzzySigmoidLayer(nn.Module):
    def __init__(self, input_dim, output_dim, top_k=None, use_owa=True):
        super(FuzzySigmoidLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.top_k = top_k
        self.use_owa = use_owa

        self.base_weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.center = nn.Parameter(torch.randn(input_dim))
        self.sharpness = nn.Parameter(torch.ones(input_dim))

        self.norm = nn.BatchNorm1d(output_dim)

    def fuzzy_membership(self, x):
        return torch.sigmoid(self.sharpness * (x - self.center))

    def owa_weighted_membership(self, μ):
        B, D = μ.size()
        top_k = self.top_k if self.top_k is not None else D

        sorted_vals, sorted_idx = torch.sort(μ, dim=1, descending=True)
        weights = torch.linspace(1.0, 0.1, steps=top_k, device=μ.device)
        weights = weights / weights.sum()
        weights = weights.unsqueeze(0).expand(B, -1)

        owa_score = torch.zeros_like(μ)
        owa_score.scatter_(1, sorted_idx[:, :top_k], weights * sorted_vals[:, :top_k])
        return owa_score

    def forward(self, x):
        μ = self.fuzzy_membership(x)  # (B, input_dim)
        μ_used = self.owa_weighted_membership(μ) if self.use_owa else μ

        B = x.size(0)
        W = self.base_weight.unsqueeze(0).expand(B, -1, -1)
        μ_exp = μ_used.unsqueeze(1)  # (B, 1, input_dim)
        fuzzy_W = W * μ_exp
        x_vec = x.unsqueeze(2)  # (B, input_dim, 1)

        out = torch.bmm(fuzzy_W, x_vec).squeeze(2) + self.bias  # (B, output_dim)
        return self.norm(F.relu(out)), μ
