"""BT/BTD model definitions.

Includes both legacy single-criterion BT/BTD and criterion-conditioned BTD.
"""

from __future__ import annotations

import torch
from torch import nn


class VectorBT(nn.Module):
    def __init__(self, num_models: int, d: int):
        super().__init__()
        self.u = nn.Embedding(num_models, d)
        self.v = nn.Embedding(num_models, d)

        nn.init.normal_(self.u.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.v.weight, mean=0.0, std=0.1)

    def forward(self, i, j, k):
        u_i = self.u(i)
        v_j = self.v(j)
        v_k = self.v(k)

        score_j = torch.sum(u_i * v_j, dim=-1)
        score_k = torch.sum(u_i * v_k, dim=-1)
        return torch.sigmoid(score_j - score_k)


class VectorBT_norm(nn.Module):
    def __init__(self, num_models: int, d: int):
        super().__init__()
        self.u = nn.Embedding(num_models, d)
        self.v = nn.Embedding(num_models, d)

        nn.init.normal_(self.u.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.v.weight, mean=0.0, std=0.1)

    def forward(self, i, j, k):
        u_i = self.u(i)
        v_j = self.v(j)
        v_k = self.v(k)

        score_j = -torch.sum((u_i - v_j) ** 2, dim=-1)
        score_k = -torch.sum((u_i - v_k) ** 2, dim=-1)
        return torch.sigmoid(score_j - score_k)


class VectorBT_bias(nn.Module):
    def __init__(self, num_models: int, d: int):
        super().__init__()
        self.u = nn.Embedding(num_models, d)
        self.v = nn.Embedding(num_models, d)
        self.b = nn.Embedding(num_models, 1)

        nn.init.normal_(self.u.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.v.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.b.weight, mean=0.0, std=0.1)

    def forward(self, i, j, k):
        u_i = self.u(i)
        v_j = self.v(j)
        v_k = self.v(k)
        b_i = self.b(i)

        score_j = torch.sum(u_i * v_j, dim=-1)
        score_k = torch.sum(u_i * v_k, dim=-1)
        return torch.sigmoid(score_j - score_k + b_i.squeeze(-1))


class VectorBTD(nn.Module):
    def __init__(self, num_models: int, d: int):
        super().__init__()
        self.u = nn.Embedding(num_models, d)
        self.v = nn.Embedding(num_models, d)
        self.log_lambda = nn.Embedding(num_models, 1)

        nn.init.normal_(self.u.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.v.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.log_lambda.weight, 0.0)

    def forward(self, i, j, k):
        u_i = self.u(i)
        v_j = self.v(j)
        v_k = self.v(k)

        score_j = torch.sum(u_i * v_j, dim=-1)
        score_k = torch.sum(u_i * v_k, dim=-1)

        log_lambda_i = self.log_lambda(i).squeeze(-1)
        tie_logit = log_lambda_i + 0.5 * (score_j + score_k)
        logits = torch.stack([tie_logit, score_j, score_k], dim=1)
        return logits


class CriteriaVectorBTD(nn.Module):
    """Criterion-conditioned BTD model from BT_criteria_2.py."""

    def __init__(self, num_criteria: int, num_models: int, d: int):
        super().__init__()
        self.u = nn.Embedding(num_criteria * num_models, d)
        self.v = nn.Embedding(num_models, d)
        self.log_lambda = nn.Embedding(num_criteria * num_models, 1)

        nn.init.normal_(self.u.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.v.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.log_lambda.weight, 0.0)

    def forward(self, c, i, j, k):
        num_models = self.v.num_embeddings
        row_idx = c * num_models + i

        u_i = self.u(row_idx)
        v_j = self.v(j)
        v_k = self.v(k)

        score_j = torch.sum(u_i * v_j, dim=-1)
        score_k = torch.sum(u_i * v_k, dim=-1)

        log_lambda_i = self.log_lambda(row_idx).squeeze(-1)
        tie_logit = log_lambda_i + 0.5 * (score_j + score_k)
        logits = torch.stack([tie_logit, score_j, score_k], dim=1)
        return logits
