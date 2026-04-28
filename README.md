# Motif-Upcycling

**Structure-preserving adaptation of pretrained Transformer models.**

This repository contains the reference implementation for the paper:

> **Motif-Upcycling: Structure-Preserving Adaptation of Transformer Models**

Motif-upcycling exposes motif-aligned structure inside selected Transformer
modules, while preserving the pretrained function at initialization. The core
implementation currently targets Qwen-style SwiGLU MLPs and supports:

- exact channel-slice factorization of gated FFN/SwiGLU blocks;
- neutral static or contextual motif routing;
- motif-local LoRA interventions;
- Scale-Aware Residual Control (SARC);
- closed-form parameter-budget utilities;
- exactness and SARC safety tests.

## Core idea

A Qwen-style SwiGLU MLP computes

```text
F(x) = W_down [SiLU(W_gate x) * (W_up x)]
```

The intermediate channel axis can be partitioned into non-overlapping slices
`S_1, ..., S_M`. Each slice defines an expert:

```text
E_m(x) = W_down[:, S_m] [SiLU(W_gate[S_m] x) * (W_up[S_m] x)]
```

Because the slices cover the intermediate axis,

```text
F(x) = sum_m E_m(x)
```

A neutral router uses

```text
alpha(x) = M * softmax(router(x))
```

with zero-initialized router output, so `alpha_m(x) = 1` at initialization.
Thus, the patched module computes the same function as the dense donor module,
up to ordinary floating-point summation differences.

## Scale-Aware Residual Control (SARC)

SARC is an optional lightweight control module. The safest variant is
adapter-only SARC:

```text
y = x + base_update + scale(ref, adapter_delta) * adapter_delta
scale = 1 + max_delta * tanh(h(r))
r = log((RMS(adapter_delta) + eps) / (RMS(ref) + eps))
```

The last layer of `h` is zero-initialized, hence `scale = 1` at initialization.
This preserves the current baseline exactly while allowing the model to learn a
bounded scale correction for the new trainable intervention.

## Installation

```bash
git clone https://github.com/kharkilirov1/motif_upcycling.git
cd motif-upcycling
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

For Hugging Face model patching, install the optional transformer dependencies:

```bash
pip install -e .[hf]
```

## Quick start

```python
from motif_upcycling import MotifUpcycleConfig, SARCConfig, patch_qwen_mlp_layers

cfg = MotifUpcycleConfig(
    num_motifs=3,
    router_hidden=128,
    router_type="contextual",
    lora_rank=8,
    lora_alpha=8.0,
    sarc=SARCConfig(enabled=True, mode="adapter_only", max_delta=0.10),
)

patch_qwen_mlp_layers(model, layer_indices=[15, 16, 17, 18], config=cfg)
```

## Local smoke tests

```bash
pytest -q
python scripts/check_exactness.py --sarc-mode off
python scripts/check_exactness.py --sarc-mode adapter_only
python scripts/count_params.py --hidden-size 2560 --intermediate-size 9728 --layers 4 --rank 8
```

Expected Qwen3-4B P4-lite parameter budget:

```text
P4 total: 3,672,076
```

Expected Qwen2.5-1.5B P4 parameter budget:

```bash
python scripts/count_params.py --hidden-size 1536 --intermediate-size 8960 --layers 4 --rank 8
```

```text
P4 total: 2,803,724
```

## Repository layout

```text
src/motif_upcycling/
  config.py       # MotifUpcycleConfig and SARCConfig
  swiglu.py       # exact SwiGLU motif wrapper
  sarc.py         # Scale-Aware Residual Control
  routers.py      # neutral static/contextual routers
  lora.py         # low-rank adapters
  patch.py        # HF Qwen patching helpers
  params.py       # closed-form parameter budgets

configs/          # example experiment configs
examples/         # usage examples
scripts/          # exactness, parameter, and CSV analysis scripts
tests/            # unit tests
paper/            # paper PDF/TeX
figures/          # publication figures
results/          # preliminary result CSVs
```

## Preliminary result snapshot

The paper reports preliminary Qwen-family evidence, including a hybrid
Qwen3.5-0.8B motif/LoRA run with approximately 0.092% trainable parameters.
These are preliminary experiments, not a universal quality claim.

| Model | Patched layers | Trainable params | Trainable share | Best mean delta |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B-Base | 10, 11 | 692,492 | 0.092% | 0.4578 |

## Limitations

This is a preliminary research framework. We do not claim optimality, universal
validity, frontier-level quality, or established long-context robustness. SARC is
implemented as an identity-preserving control mechanism; its empirical benefit
should be verified through ablations.

## Citation

```bibtex
@misc{motifupcycling2026,
  title={Motif-Upcycling: Structure-Preserving Adaptation of Transformer Models},
  author={Anonymous},
  year={2026},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
