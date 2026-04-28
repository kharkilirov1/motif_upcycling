# Theory notes

## Exact SwiGLU factorization

For a Qwen-style SwiGLU MLP,

```text
F(x) = W_down [SiLU(W_gate x) * (W_up x)]
```

split the intermediate axis into non-overlapping channel sets `S_m` whose union is
all intermediate channels. Define

```text
E_m(x) = W_down[:, S_m] [SiLU(W_gate[S_m] x) * (W_up[S_m] x)]
```

Then

```text
F(x) = sum_m E_m(x)
```

up to the single shared down bias, which is added once after summing.

## Neutral routing

With `M` motifs,

```text
alpha(x) = M * softmax(router(x))
```

and a zero-initialized router output, `router(x)=0`, hence every coefficient is
`alpha_m=1`. Therefore the routed module is function-preserving at initialization.

## SARC

The identity-preserving SARC correction is

```text
scale = 1 + lambda * tanh(h(r))
r = log((RMS(update)+eps)/(RMS(ref)+eps))
```

where the final layer of `h` is zero-initialized. At initialization, `scale=1`.
The deviation from the current baseline is bounded by `lambda * ||update||`.

## Adapter-only SARC

For pretrained models, the safest variant leaves the donor/base update unchanged:

```text
y = x + base_update + scale(ref, adapter_delta) * adapter_delta
```

This preserves the pretrained residual geometry and only controls new trainable
interventions.
