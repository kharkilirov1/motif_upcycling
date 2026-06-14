from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from .data import AOTExample, AOTVocab, OP_NONE, batch_examples, build_examples, make_vocab_for_splits
from .models import AOTStructuredComposer, RandomMoERecurrent, TinyTransformerClassifier
from .ops import FLAT_OUT, MAX_OPS, check_associative, D4_TABLE, C8_TABLE, BOOL_TABLE, soft_compose_from_onehots, compose_sequence


def avg(xs: Sequence[float]) -> float:
    return float(sum(float(x) for x in xs) / max(1, len(xs)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def iter_batches(examples: Sequence[AOTExample], vocab: AOTVocab, batch_size: int, max_len: int,
                 shuffle: bool = True, seed: int = 0):
    idx = list(range(len(examples)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch = [examples[i] for i in idx[start:start + batch_size]]
        yield batch_examples(batch, vocab, max_len)


def structured_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                    w_kind: float = 1.0, w_op: float = 1.0, w_task: float = 0.7, w_product: float = 1.0) -> torch.Tensor:
    loss = w_product * F.nll_loss(out["flat_logits"], batch["flat_target"])
    loss = loss + w_task * F.cross_entropy(out["task_logits"], batch["target_kind"])
    loss = loss + w_kind * F.cross_entropy(out["kind_logits"].reshape(-1, 4), batch["kind_labels"].reshape(-1))
    mask = batch["op_labels"].reshape(-1).ne(OP_NONE)
    if bool(mask.any()):
        loss = loss + w_op * F.cross_entropy(out["op_logits"].reshape(-1, MAX_OPS)[mask], batch["op_labels"].reshape(-1)[mask])
    return loss


def train_structured(train: Sequence[AOTExample], vocab: AOTVocab, max_len: int, epochs: int = 80,
                     batch_size: int = 64, lr: float = 2e-3, seed: int = 0) -> Tuple[AOTStructuredComposer, List[float]]:
    set_seed(seed)
    model = AOTStructuredComposer(len(vocab), d_model=64, max_len=max_len)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    losses: List[float] = []
    for ep in range(epochs):
        model.train()
        ep_losses = []
        for b in iter_batches(train, vocab, batch_size, max_len, shuffle=True, seed=seed + ep):
            out = model(b["input_ids"])
            loss = structured_loss(out, b)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            ep_losses.append(float(loss.detach()))
        losses.append(avg(ep_losses))
    return model, losses


def train_classifier(model: torch.nn.Module, train: Sequence[AOTExample], vocab: AOTVocab, max_len: int,
                     epochs: int = 80, batch_size: int = 64, lr: float = 2e-3, seed: int = 0) -> Tuple[torch.nn.Module, List[float]]:
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    losses: List[float] = []
    for ep in range(epochs):
        model.train()
        ep_losses = []
        for b in iter_batches(train, vocab, batch_size, max_len, shuffle=True, seed=seed + ep):
            logits = model(b["input_ids"])
            loss = F.cross_entropy(logits, b["flat_target"])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            ep_losses.append(float(loss.detach()))
        losses.append(avg(ep_losses))
    return model, losses


@torch.no_grad()
def eval_structured(model: AOTStructuredComposer, examples: Sequence[AOTExample], vocab: AOTVocab, max_len: int,
                    batch_size: int = 128) -> Dict[str, Any]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    soft_correct: List[bool] = []
    hard_correct: List[bool] = []
    task_correct: List[bool] = []
    kind_correct: List[bool] = []
    op_correct: List[bool] = []
    for b in iter_batches(examples, vocab, batch_size, max_len, shuffle=False):
        out = model(b["input_ids"])
        soft = out["flat_logits"].argmax(-1)
        hard = model.hard_predict(b["input_ids"])
        soft_correct.extend((soft == b["flat_target"]).tolist())
        hard_correct.extend((hard["flat_pred"] == b["flat_target"]).tolist())
        task_correct.extend((out["task_logits"].argmax(-1) == b["target_kind"]).tolist())
        nonpad = b["input_ids"].reshape(-1).ne(0)
        kind_correct.extend((out["kind_logits"].argmax(-1).reshape(-1)[nonpad] == b["kind_labels"].reshape(-1)[nonpad]).tolist())
        op_mask = b["op_labels"].reshape(-1).ne(OP_NONE)
        if bool(op_mask.any()):
            op_correct.extend((out["op_logits"].argmax(-1).reshape(-1)[op_mask] == b["op_labels"].reshape(-1)[op_mask]).tolist())
    metrics: Dict[str, Any] = {
        "soft_product_acc": avg(soft_correct),
        "hard_product_acc": avg(hard_correct),
        "task_kind_acc": avg(task_correct),
        "token_kind_acc_nonpad": avg(kind_correct),
        "operator_token_acc": avg(op_correct),
    }
    # By length and domain.
    for L in sorted({e.seq_len for e in examples}):
        sub = [e for e in examples if e.seq_len == L]
        b = batch_examples(sub, vocab, max_len)
        out = model(b["input_ids"])
        hp = model.hard_predict(b["input_ids"])["flat_pred"]
        metrics[f"len_{L}_hard_acc"] = avg((hp == b["flat_target"]).tolist())
        metrics[f"len_{L}_soft_acc"] = avg((out["flat_logits"].argmax(-1) == b["flat_target"]).tolist())
    for dom in sorted({e.target_domain for e in examples}):
        sub = [e for e in examples if e.target_domain == dom]
        b = batch_examples(sub, vocab, max_len)
        hp = model.hard_predict(b["input_ids"])["flat_pred"]
        metrics[f"domain_{dom}_hard_acc"] = avg((hp == b["flat_target"]).tolist())
    return metrics


@torch.no_grad()
def eval_classifier(model: torch.nn.Module, examples: Sequence[AOTExample], vocab: AOTVocab, max_len: int,
                    batch_size: int = 128) -> Dict[str, Any]:
    model.eval()
    preds: List[int] = []
    targets: List[int] = []
    for b in iter_batches(examples, vocab, batch_size, max_len, shuffle=False):
        pred = model(b["input_ids"]).argmax(-1)
        preds.extend([int(x) for x in pred.tolist()])
        targets.extend([int(x) for x in b["flat_target"].tolist()])
    metrics: Dict[str, Any] = {"product_acc": avg([p == y for p, y in zip(preds, targets)])}
    for L in sorted({e.seq_len for e in examples}):
        idx = [i for i, e in enumerate(examples) if e.seq_len == L]
        metrics[f"len_{L}_acc"] = avg([preds[i] == targets[i] for i in idx])
    for dom in sorted({e.target_domain for e in examples}):
        idx = [i for i, e in enumerate(examples) if e.target_domain == dom]
        metrics[f"domain_{dom}_acc"] = avg([preds[i] == targets[i] for i in idx])
    return metrics


def run_smoke_experiment(
    seed: int = 10,
    train_lengths: Sequence[int] = (1, 2, 3, 4, 5),
    test_lengths: Sequence[int] = (5, 10, 20, 50),
    train_per_len: int = 96,
    test_per_len: int = 48,
    aot_epochs: int = 80,
    baseline_epochs: int = 60,
    batch_size: int = 64,
    distractor_rate: float = 0.35,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    set_seed(seed)
    train = build_examples(train_lengths, train_per_len, seed=seed, distractor_rate=distractor_rate, alias_mode="mixed")
    test = build_examples(test_lengths, test_per_len, seed=seed + 99, distractor_rate=distractor_rate, alias_mode="mixed")
    vocab = make_vocab_for_splits(train, test)
    max_len = min(256, max(len(vocab.tokens(e.text)) for e in train + test))

    aot, aot_losses = train_structured(train, vocab, max_len, epochs=aot_epochs, batch_size=batch_size, seed=seed)
    transformer = TinyTransformerClassifier(len(vocab), d_model=64, max_len=max_len)
    transformer, tr_losses = train_classifier(transformer, train, vocab, max_len, epochs=baseline_epochs, batch_size=batch_size, seed=seed + 1)
    moe = RandomMoERecurrent(len(vocab), d_model=64, max_len=max_len)
    moe, moe_losses = train_classifier(moe, train, vocab, max_len, epochs=baseline_epochs, batch_size=batch_size, seed=seed + 2)

    report: Dict[str, Any] = {
        "version": "aot_v8_structured_operator_transformer",
        "setup": {
            "train_lengths": list(train_lengths),
            "test_lengths": list(test_lengths),
            "train_examples": len(train),
            "test_examples": len(test),
            "vocab_size": len(vocab),
            "max_len": max_len,
            "distractor_rate": distractor_rate,
            "aot_epochs": aot_epochs,
            "baseline_epochs": baseline_epochs,
        },
        "unit_properties": {
            "d4_associative": check_associative(D4_TABLE),
            "c8_associative": check_associative(C8_TABLE),
            "bool_associative": check_associative(BOOL_TABLE),
            "soft_d4_matches_hard_sample": soft_compose_from_onehots([1, 4, 2, 6], D4_TABLE) == compose_sequence(D4_TABLE, [1, 4, 2, 6]),
        },
        "loss_tail": {
            "aot_last": aot_losses[-5:],
            "transformer_last": tr_losses[-5:],
            "random_moe_last": moe_losses[-5:],
        },
        "train_metrics": {
            "aot": eval_structured(aot, train, vocab, max_len),
            "transformer": eval_classifier(transformer, train, vocab, max_len),
            "random_moe": eval_classifier(moe, train, vocab, max_len),
        },
        "test_metrics": {
            "aot": eval_structured(aot, test, vocab, max_len),
            "transformer": eval_classifier(transformer, test, vocab, max_len),
            "random_moe": eval_classifier(moe, test, vocab, max_len),
        },
    }

    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "aot_v8_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_markdown_report(report, out / "aot_v8_report.md")
        write_csv_summary(report, out / "aot_v8_summary.csv")
    return report


def write_markdown_report(report: Dict[str, Any], path: Path) -> None:
    tm = report["test_metrics"]
    lines = []
    lines.append("# AOT v8 Structured Operator Transformer Report\n")
    lines.append("## Setup\n")
    for k, v in report["setup"].items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("\n## Unit properties\n")
    for k, v in report["unit_properties"].items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("\n## Main test metrics\n")
    lines.append("| model | overall | len_5 | len_10 | len_20 | len_50 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model_name, metrics in tm.items():
        if model_name == "aot":
            overall = metrics.get("hard_product_acc", 0.0)
            prefix = "hard"
        else:
            overall = metrics.get("product_acc", 0.0)
            prefix = ""
        vals = []
        for L in [5, 10, 20, 50]:
            vals.append(metrics.get(f"len_{L}_{prefix + '_' if prefix else ''}acc", metrics.get(f"len_{L}_acc", 0.0)))
        lines.append(f"| {model_name} | {overall:.3f} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    lines.append("\n## AOT parser metrics\n")
    aot = tm["aot"]
    for k in ["task_kind_acc", "token_kind_acc_nonpad", "operator_token_acc", "soft_product_acc", "hard_product_acc"]:
        lines.append(f"- **{k}**: `{aot.get(k)}`")
    lines.append("\n## Interpretation\n")
    lines.append("Structured AOT uses explicit finite operator tables for D4, C8, and Boolean unary composition. ")
    lines.append("The baselines must learn the composition rule from sequence-level labels only. The intended signal is length generalization: training is on short chains, evaluation includes longer chains.")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv_summary(report: Dict[str, Any], path: Path) -> None:
    rows = []
    for split in ["train_metrics", "test_metrics"]:
        for model, metrics in report[split].items():
            overall_key = "hard_product_acc" if model == "aot" else "product_acc"
            rows.append({"split": split.replace("_metrics", ""), "model": model, "metric": "overall_acc", "value": metrics.get(overall_key)})
            for k, v in metrics.items():
                if k.startswith("len_") or k.startswith("domain_"):
                    rows.append({"split": split.replace("_metrics", ""), "model": model, "metric": k, "value": v})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "model", "metric", "value"])
        w.writeheader(); w.writerows(rows)
