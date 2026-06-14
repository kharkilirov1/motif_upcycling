from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import (
    BOOL_TABLE, C8_TABLE, D4_TABLE, DOMAIN_SIZES, DOMAIN_TABLES, FLAT_OUT,
    MAX_OPS, soft_compose_step,
)


class MaskedMean(nn.Module):
    def forward(self, h: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.ne(0).float().unsqueeze(-1)
        return (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)


class TinyTokenEncoder(nn.Module):
    """Small token encoder used by the AOT parser.

    It is intentionally light: the main experiment is whether explicit structured
    composition generalizes once operators are parsed.
    """
    def __init__(self, vocab_size: int, d_model: int = 64, max_len: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(max_len, d_model) * 0.01)
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        t = input_ids.shape[1]
        h = self.embedding(input_ids) + self.pos[:t].unsqueeze(0)
        h = self.proj(h)
        return h.masked_fill(input_ids.eq(0).unsqueeze(-1), 0.0)


def transition_tensor(table: Sequence[Sequence[int]]) -> torch.Tensor:
    n = len(table)
    t = torch.zeros(n, n, n)
    for op in range(n):
        for state in range(n):
            t[op, state, int(table[op][state])] = 1.0
    return t


class AOTStructuredComposer(nn.Module):
    """Architectural Operator Transformer core.

    Components:
      - token perception / parser
      - router: kind_head chooses none/d4/c8/bool
      - structured experts: exact finite composition tables
      - task selector: chooses which domain's state should be read out

    The final output is a flat distribution over d4[0:8], c8[8:16], bool[16:20].
    """
    def __init__(self, vocab_size: int, d_model: int = 64, max_len: int = 256, temperature: float = 1.0):
        super().__init__()
        self.encoder = TinyTokenEncoder(vocab_size, d_model=d_model, max_len=max_len)
        self.pool = MaskedMean()
        self.kind_head = nn.Linear(d_model, 4)      # none, d4, c8, bool
        self.op_head = nn.Linear(d_model, MAX_OPS)  # domain-local op id; bool uses first 4
        self.task_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3))
        self.temperature = temperature
        self.register_buffer("d4_transition", transition_tensor(D4_TABLE), persistent=False)
        self.register_buffer("c8_transition", transition_tensor(C8_TABLE), persistent=False)
        self.register_buffer("bool_transition", transition_tensor(BOOL_TABLE), persistent=False)

    def _compose_domain(self, kind_probs: torch.Tensor, op_probs_full: torch.Tensor,
                        domain_kind_id: int, n_ops: int, transition: torch.Tensor) -> torch.Tensor:
        bsz, t, _ = kind_probs.shape
        state = kind_probs.new_zeros(bsz, n_ops)
        state[:, 0] = 1.0
        trans = transition.to(dtype=state.dtype, device=state.device)
        for pos in range(t):
            p_domain = kind_probs[:, pos, domain_kind_id].unsqueeze(-1)  # [B,1]
            p_ops = op_probs_full[:, pos, :n_ops]
            # If token is not this domain, compose identity. If it is this domain,
            # compose the predicted operator distribution.
            effective = p_domain * p_ops
            effective = effective.clone()
            effective[:, 0] = effective[:, 0] + (1.0 - p_domain.squeeze(-1))
            effective = effective / effective.sum(-1, keepdim=True).clamp_min(1e-8)
            # out[b,n] = sum_o sum_s p(o) p(s) 1[table[o][s] = n]
            state = torch.einsum("bo,bs,osn->bn", effective, state, trans)
        return state

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(input_ids)
        kind_logits = self.kind_head(h)
        op_logits = self.op_head(h)
        pooled = self.pool(h, input_ids)
        task_logits = self.task_head(pooled)
        kind_probs = F.softmax(kind_logits / self.temperature, dim=-1)
        op_probs = F.softmax(op_logits / self.temperature, dim=-1)

        d4_state = self._compose_domain(kind_probs, op_probs, 1, 8, self.d4_transition)
        c8_state = self._compose_domain(kind_probs, op_probs, 2, 8, self.c8_transition)
        bool_state = self._compose_domain(kind_probs, op_probs, 3, 4, self.bool_transition)
        task_probs = F.softmax(task_logits, dim=-1)

        flat_probs = input_ids.new_zeros(input_ids.shape[0], FLAT_OUT).float()
        flat_probs[:, 0:8] = task_probs[:, 0:1] * d4_state
        flat_probs[:, 8:16] = task_probs[:, 1:2] * c8_state
        flat_probs[:, 16:20] = task_probs[:, 2:3] * bool_state
        flat_logits = torch.log(flat_probs.clamp_min(1e-9))
        return {
            "kind_logits": kind_logits,
            "op_logits": op_logits,
            "task_logits": task_logits,
            "d4_state": d4_state,
            "c8_state": c8_state,
            "bool_state": bool_state,
            "flat_logits": flat_logits,
        }

    @torch.no_grad()
    def hard_predict(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.forward(input_ids)
        kind_pred = out["kind_logits"].argmax(-1)
        op_pred = out["op_logits"].argmax(-1)
        task_pred = out["task_logits"].argmax(-1)
        bsz, t = input_ids.shape
        flat = []
        domain_local = []
        for b in range(bsz):
            states = {"d4": 0, "c8": 0, "bool": 0}
            for pos in range(t):
                k = int(kind_pred[b, pos])
                o = int(op_pred[b, pos])
                if k == 1 and o < 8:
                    states["d4"] = int(D4_TABLE[o][states["d4"]])
                elif k == 2 and o < 8:
                    states["c8"] = int(C8_TABLE[o][states["c8"]])
                elif k == 3 and o < 4:
                    states["bool"] = int(BOOL_TABLE[o][states["bool"]])
            tk = int(task_pred[b])
            if tk == 0:
                flat.append(states["d4"]); domain_local.append(states["d4"])
            elif tk == 1:
                flat.append(8 + states["c8"]); domain_local.append(states["c8"])
            else:
                flat.append(16 + states["bool"]); domain_local.append(states["bool"])
        return {
            "flat_pred": torch.tensor(flat, device=input_ids.device, dtype=torch.long),
            "domain_local_pred": torch.tensor(domain_local, device=input_ids.device, dtype=torch.long),
            "kind_pred": kind_pred,
            "op_pred": op_pred,
            "task_pred": task_pred,
        }


class TinyTransformerClassifier(nn.Module):
    """Transformer-only sequence classifier baseline."""
    def __init__(self, vocab_size: int, d_model: int = 64, nhead: int = 4, num_layers: int = 1,
                 max_len: int = 256, out_dim: int = FLAT_OUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(max_len, d_model) * 0.01)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = MaskedMean()
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, out_dim))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        t = input_ids.shape[1]
        h = self.embedding(input_ids) + self.pos[:t].unsqueeze(0)
        pad_mask = input_ids.eq(0)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return self.out(self.pool(h, input_ids))


class RandomMoERecurrent(nn.Module):
    """Unstructured MoE composer baseline.

    It has a router and several MLP experts, but no algebraic composition table.
    This is meant to test whether named structured experts help length-generalization.
    """
    def __init__(self, vocab_size: int, d_model: int = 64, n_experts: int = 4, max_len: int = 256,
                 out_dim: int = FLAT_OUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(max_len, d_model) * 0.01)
        self.router = nn.Linear(d_model * 2, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Tanh(), nn.Linear(d_model, d_model))
            for _ in range(n_experts)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, out_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, t = input_ids.shape
        x = self.embedding(input_ids) + self.pos[:t].unsqueeze(0)
        state = x.new_zeros(bsz, x.shape[-1])
        for pos in range(t):
            emb = x[:, pos]
            active = input_ids[:, pos].ne(0).float().unsqueeze(-1)
            inp = torch.cat([state, emb], -1)
            weights = F.softmax(self.router(inp), -1)
            updates = torch.stack([expert(inp) for expert in self.experts], dim=1)  # [B,E,D]
            delta = (weights.unsqueeze(-1) * updates).sum(1)
            state = self.norm(state + active * delta)
        return self.out(state)
