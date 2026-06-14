from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .ops import (
    BOOL_OP2ID, BOOL_OPS, C8_OP2ID, C8_OPS, D4_OP2ID, D4_OPS,
    DOMAIN2ID, DOMAIN_CLASS, DOMAIN_OPS, DOMAIN_SIZES, DOMAIN_TABLES,
    FLAT_OFFSETS, FLAT_OUT, MAX_OPS, compose_sequence,
)

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
KIND_NONE = 0
OP_NONE = MAX_OPS

FILLERS = [
    "please", "compose", "apply", "sequence", "chain", "then", "next",
    "and", "after", "start", "stop", "result", "ignore", "noise", "with",
]
DOMAIN_ALIASES = {
    "d4": ["d4", "spatial", "square"],
    "c8": ["c8", "cyclic", "modular"],
    "bool": ["bool", "logic", "boolean"],
}
OP_ALIASES = {
    "d4": {
        "id": ["id", "stay", "identity"],
        "rot90": ["rot90", "turn_right", "quarter_turn"],
        "rot180": ["rot180", "half_turn"],
        "rot270": ["rot270", "turn_left"],
        "flip_h": ["flip_h", "mirror_lr"],
        "flip_v": ["flip_v", "mirror_ud"],
        "flip_diag": ["flip_diag", "diag_main"],
        "flip_anti_diag": ["flip_anti_diag", "diag_anti"],
    },
    "c8": {f"add{i}": [f"add{i}", f"plus{i}", f"shift{i}"] for i in range(8)},
    "bool": {
        "keep": ["keep", "same", "bool_id"],
        "invert": ["invert", "not", "negate"],
        "false": ["false", "zero", "always_false"],
        "true": ["true", "one", "always_true"],
    },
}


def canonical_op_id(domain: str, token: str) -> Optional[int]:
    token = token.lower()
    for canon, aliases in OP_ALIASES[domain].items():
        if token in aliases:
            if domain == "d4": return D4_OP2ID[canon]
            if domain == "c8": return C8_OP2ID[canon]
            if domain == "bool": return BOOL_OP2ID[canon]
    return None


def token_kind_op(token: str) -> Tuple[int, int]:
    for domain in ["d4", "c8", "bool"]:
        op = canonical_op_id(domain, token)
        if op is not None:
            return DOMAIN2ID[domain], op
    return KIND_NONE, OP_NONE


@dataclass(frozen=True)
class AOTExample:
    text: str
    target_domain: str
    target_kind: int          # 0=d4,1=c8,2=bool for task head
    flat_target: int          # 0..19
    target_op: int            # domain-local target op id
    seq_len: int              # number of target-domain ops, not raw token length
    op_counts: Dict[str, int]


class AOTVocab:
    def __init__(self, texts: Iterable[str]):
        toks = sorted({tok for text in texts for tok in self.tokens(text)})
        # Add all aliases to make train/test generation stable.
        extra = set(FILLERS)
        for aliases in DOMAIN_ALIASES.values(): extra.update(aliases)
        for dom in OP_ALIASES:
            for aliases in OP_ALIASES[dom].values(): extra.update(aliases)
        toks = sorted(set(toks) | extra)
        self.itos = ["<pad>", "<unk>"] + toks
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def tokens(self, text: str) -> List[str]:
        return TOKEN_RE.findall(text.lower())

    def encode(self, text: str, max_len: int) -> List[int]:
        ids = [self.stoi.get(t, 1) for t in self.tokens(text)[:max_len]]
        return ids + [0] * max(0, max_len - len(ids))

    def token_labels(self, text: str, max_len: int) -> Tuple[List[int], List[int]]:
        kinds: List[int] = []
        ops: List[int] = []
        for tok in self.tokens(text)[:max_len]:
            k, o = token_kind_op(tok)
            kinds.append(k)
            ops.append(o)
        while len(kinds) < max_len:
            kinds.append(KIND_NONE); ops.append(OP_NONE)
        return kinds, ops


def _choose_alias(rng: random.Random, domain: str, canonical: str, alias_mode: str = "mixed") -> str:
    aliases = OP_ALIASES[domain][canonical]
    if alias_mode == "canonical":
        return aliases[0]
    if alias_mode == "heldout":
        return aliases[-1]
    return rng.choice(aliases)


def build_example(seq_len: int, rng: random.Random, target_domain: Optional[str] = None,
                  distractor_rate: float = 0.35, alias_mode: str = "mixed") -> AOTExample:
    if target_domain is None:
        target_domain = rng.choice(["d4", "c8", "bool"])
    target_ops: List[int] = []
    op_counts = {"d4": 0, "c8": 0, "bool": 0}
    stream: List[Tuple[str, int]] = []  # (domain, op id)

    for _ in range(seq_len):
        dom = target_domain
        op_id = rng.randrange(DOMAIN_SIZES[dom])
        stream.append((dom, op_id))
        target_ops.append(op_id)
        op_counts[dom] += 1
        # Add 0-2 distractor ops from other domains.
        if rng.random() < distractor_rate:
            for _ in range(1 + int(rng.random() < 0.25)):
                other = rng.choice([d for d in ["d4", "c8", "bool"] if d != target_domain])
                oid = rng.randrange(DOMAIN_SIZES[other])
                stream.append((other, oid)); op_counts[other] += 1

    rng.shuffle(stream) if False else None  # Keep order; composition is order-sensitive.

    target_op = compose_sequence(DOMAIN_TABLES[target_domain], target_ops)
    flat_target = FLAT_OFFSETS[target_domain] + target_op
    target_kind = DOMAIN_CLASS[target_domain]

    # Surface text: domain instruction + noisy operator stream.
    domain_word = rng.choice(DOMAIN_ALIASES[target_domain])
    tokens = [rng.choice(["task", "please", "compose"]), domain_word, rng.choice(["start", "apply", "chain"])]
    for idx, (dom, op_id) in enumerate(stream):
        if rng.random() < 0.25:
            tokens.append(rng.choice(["then", "next", "and", "after"]));
        canon = DOMAIN_OPS[dom][op_id]
        tokens.append(_choose_alias(rng, dom, canon, alias_mode))
        if rng.random() < 0.08:
            tokens.append(rng.choice(["noise", "ignore", "with"]))
    tokens.append(rng.choice(["stop", "result", "end"]))
    return AOTExample(
        text=" ".join(tokens),
        target_domain=target_domain,
        target_kind=target_kind,
        flat_target=flat_target,
        target_op=target_op,
        seq_len=seq_len,
        op_counts=op_counts,
    )


def build_examples(seq_lens: Sequence[int], count_per_len: int, seed: int,
                   target_domains: Optional[Sequence[str]] = None,
                   distractor_rate: float = 0.35, alias_mode: str = "mixed") -> List[AOTExample]:
    rng = random.Random(seed)
    out: List[AOTExample] = []
    for L in seq_lens:
        for _ in range(count_per_len):
            dom = rng.choice(list(target_domains)) if target_domains else None
            out.append(build_example(L, rng, dom, distractor_rate, alias_mode))
    rng.shuffle(out)
    return out


def make_vocab_for_splits(*splits: Sequence[AOTExample]) -> AOTVocab:
    return AOTVocab([e.text for split in splits for e in split])


def batch_examples(examples: Sequence[AOTExample], vocab: AOTVocab, max_len: Optional[int] = None) -> Dict[str, torch.Tensor]:
    if max_len is None:
        max_len = max(len(vocab.tokens(e.text)) for e in examples)
    input_ids = []
    kind_labels = []
    op_labels = []
    for e in examples:
        input_ids.append(vocab.encode(e.text, max_len))
        k, o = vocab.token_labels(e.text, max_len)
        kind_labels.append(k); op_labels.append(o)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "kind_labels": torch.tensor(kind_labels, dtype=torch.long),
        "op_labels": torch.tensor(op_labels, dtype=torch.long),
        "target_kind": torch.tensor([e.target_kind for e in examples], dtype=torch.long),
        "flat_target": torch.tensor([e.flat_target for e in examples], dtype=torch.long),
        "target_op": torch.tensor([e.target_op for e in examples], dtype=torch.long),
        "seq_len": torch.tensor([e.seq_len for e in examples], dtype=torch.long),
    }
