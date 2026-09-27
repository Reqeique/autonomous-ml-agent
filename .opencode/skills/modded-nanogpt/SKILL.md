---
name: modded-nanogpt
description: Provably-fast techniques for training small autoregressive transformers (<2B params) to lower token-level cross entropy. Distilled from KellerJordan/modded-nanogpt (5.6k stars), the competitive NanoGPT speedrun that drove GPT-2 124M training time from 45min to ~1.2min on 8xH100. Use when choosing optimizer, architecture tweaks, LR schedule, attention pattern, or numeric precision for a char/token-level AR LM where we care about val cross-entropy (and thus BPC) per training FLOP.
license: MIT
source: https://github.com/KellerJordan/modded-nanogpt
---

# Modded-NanoGPT — Small-AR-Transformer Training Tricks

This skill distills the techniques that survived competitive peer review in the modded-nanogpt speedrun (89 records as of 2026-07). The benchmark task is "train GPT-2 124M to val loss ≤ 3.28 on FineWeb using 8x H100" — directly analogous to our setting (small AR transformer, val cross-entropy target, fixed training token budget). **Translated to BPC: their 3.28 nats target on BPE tokens corresponds to ~1.37 BPC-equivalent density; our enwik8 target is ≤1.05 BPC on raw bytes.** The levers they pulled are mostly transferable; the decoder-only GPT-2 architecture and our PerPos chunk-decoder diverge only at the head.

Treat each technique as a hypothesis to test on our run, not a guarantee. Every entry below is load-bearing for some record in their benchmark; most were independently re-verified via `p < 0.01` val-loss significance tests across multiple seeds.

## When to Use This Skill

- Choosing an optimizer for a small transformer (Muon vs AdamW vs hybrid).
- Tuning LR schedule, batch size schedule, or warmup for a fixed-token-budget run.
- Deciding architectural tweaks: rotary vs absolute pos, logit softcap, untied embed/head, value embeddings, ReLU² MLP, QK-norm.
- Choosing numeric precision: FP8 matmul weights, BF16 activations, mixed-precision optimizer state.
- Picking attention structure: short-window vs long-window warmup, sparse attention gates, FlexAttention, paired-head attention.
- Deciding whether multi-token prediction auxiliary losses help cross-entropy on a held-out val set.
- Designing skip connections / residual structure for very shallow stacks (U-Net, MUDD, hyper-connections).

**Not in scope here:** inference, serving, deployment, quantization-to-deploy. This skill is strictly about *training-time* decisions that move val cross-entropy down per training FLOP.

## Anchors to Our Setting

| Dimension | NanoGPT speedrun | Our enwik8 PerPos run |
|---|---|---|
| Architecture | Decoder-only GPT-2, 124M-350M params | PerPos Hierarchical Char Transformer, byte vocab=257, d_model=512, 6 chunk dec layers |
| Tokenization | BPE (FineWeb, 10M tokens val) | Byte-level UTF-8 (enwik8, 10M chars subset) |
| Target metric | Val CE ≤ 3.28 nats | Val CE → BPC ≤ 1.05 (i.e. CE/ln(2) ≤ 1.05) |
| Current best | ~1.5 min on 8xH100 to reach target | BPC ≈ 3.49 (val_loss=2.4216 nats); all recent runs crashed at epoch ≤3 |
| Hardware | 8x H100 (multi-GPU, NCCL) | Kaggle TPU v3-8 (8 TPU cores, TF/Keras) |
| Throughput lever | Mixed FP8 matmul, Triton kernels, Muon | XLA compilation on TPU, bf16, optional half-precision |

The TPU/Keras gap is real: most of the speedrun's wallclock wins are CUDA-specific (FP8 `_scaled_mm`, Triton kernels, FlashAttention-3). What transfers cleanly across to TPU are the **optimizer, architecture, and schedule choices**, not the CUDA micro-kernels. The sections below are ordered by what transfers most cleanly.

## 1. Optimizer — Muon (Highest-Leverage Transfer)

Muon is the single biggest non-architecture win in the speedrun, responsible for ~50% of the time-to-target improvement over AdamW. It is applicable to any 2D parameter (weights of Linear/Embed), not to embeddings/layernorms/scalars.

**Definition (from modded-nanogpt `train_gpt.py`):**

```python
@torch.compile
def zeroth_power_via_newtonschulz5(G, steps=5, eps=1e-7):
    # Non-convergent quintic; spectral norm ≤ 1.13 after 5 iters
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)
```

The full update is: `w -= lr * newton_schulz5(momentum_buffer)` after Nesterov momentum update on the gradient.

**Why it works for us:**
- ~1.5x better sample-efficiency than AdamW on nanoGPT.
- Lower memory than Adam (one buffer vs two).
- <2% wallclock overhead per step, amortized across epochs.

**What to port on TPU:**
- The Newton-Schulz iteration is pure matmul/elementwise — expressible in JAX/TF without Triton.
- Skip FP8 paths; use BF16 for the orthogonalization.
- Apply Muon to all 2D weights in the transformer (Q/K/V/O, MLP up/down, embedding). Keep Adam for scalars, LN gains, embeddings if desired.

**2025-2026 Muon variants the speedrun discovered (in priority order):**

1. **Polar Express** (record 38, arXiv 2505.16932) — replaces Newton-Schulz with a faster-converging 5-iteration polar decomposition. Same memory, fewer steps for the same spectral-norm bound.
2. **NorMuon** (record 41, arXiv 2510.05491) — normalizes the gradient before orthogonalization, helping Muon stability under distribution shift across the training run.
3. **Mixed-precision Muon** (record 57) — runs orthogonalization in BF16, Adam-state in FP32 for the small-parameter banks.
4. **Paired-head Muon** (record 80) — orthogonize Q and K in pairs per-head instead of across the full Q/K matrix. Marginal win at small head dim.

**Caveat for our run:** The Muon LR is **not** the AdamW LR — speedrun uses Muon LR ~4e-3 with a cosine schedule that decays to 0.1 of the peak, not 0. The peak LR tends to be 2-4x the equivalent AdamW LR for the same parameter count.

## 2. Architectural Tweaks (Transferable, Independent of Optimizer)

Each of these is a record-setter in the speedrun; each is a *hypothesis* for our PerPos transformer, not a guarantee. Order roughly by leverage vs integration cost.

### Rotary embeddings (record 2)
- Replace absolute sinusoidal with RoPE. Standard practice now.
- Universal in all subsequent records; no record has gone back.

### ReLU² MLP (record 5)
- Replace GELU with `ReLU(x @ W1.T)^2 @ W2.T`.
- ~1-2% val-loss improvement vs GELU at fixed FLOPs in the speedrun.
- Easy port to JAX/TF.

### QK-norm (record 5)
- LayerNorm on Q and K before the dot product.
- Stabilizes attention at large sequence length; cheap.

### Logit softcap (record 5, then tuned in records 18, 54)
- `logits = softcap * tanh(logits / softcap)`.
- Started at softcap=30, lowered to 15 (record 18), then asymmetric rescale (record 54).
- Prevents occasional logit blow-ups that destabilize cross-entropy gradients.
- **Direct port to our char-level output head** — our vocab=257 means logit scale matters more, not less.

### Untied embedding and head (record 8)
- Use two separate matrices for input embedding and output head, not a single shared one.
- ~8% speedup at matched val-loss, no extra FLOPs upstream.

### Value-embedding / value-skip connections (records 9, 14, 55)
- An auxiliary value-embedding pathway that skips past attention layers and adds back later.
- The U-net / skip-connection pattern (records 9, 11, 45, 50) is a *family* — the speedrun kept refining how the skip is gated: no-gate → fixed lambda → learnable gate → gate-on-value-embeds-only.
- **Record 55 (gated value-skip) is a direct candidate for our PerPos chunk-decoder** — we already have a compressor + PerPos chunk decoder; a gated value-skip from compressor output down to chunk-decoder output might compound well with the hierarchical structure.

### Drop-first-MLP / drop-first-attn (records 30, 35)
- The first MLP and the first attention layer are redundant with the embedding; dropping one of them costs ~0 loss and saves FLOPs.
- **Candidate for our chunk-decoder**: if the first chunk decoder layer is just resampling the compressor output, it may be droppable.

### Bigram-hash embedding (record 62)
- Hash-trick bigram lookup added to the byte embedding.
- Cheap win at small vocab; our byte vocab=257 makes this especially cheap.
- **Direct port candidate** for the input embedding.

### Multi-token prediction auxiliary loss (records 53, 60, 88)
- Predict not just next token but also the token 2/3 positions ahead, with a separate softcapped cross-entropy.
- Used by records 53+ across the entire AR family; no regression reported.
- **For our char-level setting** this is a natural regularizer — n-gram-style supervision baked into the loss.

### Paired-head attention (record 58)
- Pair adjacent heads, share K/V across the pair.
- ~5% speedup at fixed loss; not a method win, a parameter win.

### Sparse attention gate (record 28)
- Replace dense causal attention with a learned sparse gate.
- Record 28 reports no significant loss change at 1024 context.

## 3. LR and Batch Schedule

The speedrun's schedule evolved substantially. The best-documented stops:

- **Peak LR**: 2-4x AdamW reference LR for the same parameter count when using Muon.
- **LR decay**: cosine decaying to 0.1 of peak (not 0). Decaying to 0 wastes the last few hundred steps.
- **Cooldown fraction**: ~45% of total steps is in cooldown (records 26, 33, 40). Aggressive — they spend a lot of time in decay.
- **Warmup**: 2% linear warmup, then rise to peak, then cosine. For Muon: momentum warmup also kicks in (record 9).
- **Batch size schedule** (record 46): start small, increase mid-training. Sample-efficiency win.
- **Attention window warmup** (record 13): start training with a short attention context (e.g. 1024 tokens) and grow to the full window over the first 30% of training. **Directly applicable** — enwik8 chunking is a natural fit.
- **Max-seq-len schedule** (record 72): like batch schedule but on sequence length. Both together compound.

## 4. Numeric Precision (Lessons That Transfer to TPU)

- **BF16 activations everywhere** (record 10) — no FP32 activation paths in the forward pass. Stable on H100; should be stable on TPU v3-8 with bf16.
- **FP8 matmul weights** (records 19, 84, 89) — `_scaled_mm` with FP8 weight storage and BF16 input/output. **TPU v3 does not have FP8** — skip this. TPU v4-lite / v5 / v6e have Int8 paths but with different semantics; benchmark before trusting.
- **BF16 cross-entropy** (record 37) — compute the softmax+CE in BF16, not FP32. Unexpectedly stable and saves ~5% of step time.
- **Mixed precision optimizer state** (record 57) — keep Adam/Muhybrid state in FP32 for scalars/embeds, BF16 for large matrices.

## 5. Attention Structure Wins

- **Short-to-long attention warmup** (records 12, 13): start at 1024 context, grow. `FlexAttention` enables variable sparsity; on TPU use `dilated_attention` or a custom block-sparse mask.
- **Sparse attention gate** (record 28): learned gate over which tokens attend — saves FLOPs while preserving loss.
- **Split value embeddings** (records 15, 16, 17): split the value-embedding into multiple sub-parameters, sparsify selectively.

## 6. What Doesn't Transfer to TPU/Keras

- `torch._scaled_mm` (CUDA FP8).
- Triton kernels (`FusedLinearReLUSquareFunction`, `FusedSoftcappedCrossEntropy`, `dc_attention_postonly_nodd_correction_add_base_triton`).
- FlashAttention-3 (FA3) — TPU has its own XLA-fused attention path; use `keras.layers.Attention` with causal masking or `dilated_attention`.
- `torch.compile` — TPU relies on `jit_compile=True` in `model.fit` / `pmap` / `spmd`.
- `torch.distributed` all-reduce overlap tricks (records 22-24) — TPU v3-8 uses `tpu_strategy.unwrap` and automatic SPMD sharding instead.

## How to Use This Skill on Our Project

**Step 1 — try Muon first.**
A Muon optimizer on the PerPos chunk decoder parameters (keeping Adam on the small token decoder and embeddings) is the highest-leverage single change. Use the Newton-Schulz-5 iteration above. Translates to ~30 lines of JAX/TF code inside `model.fit`'s `train_step`.

**Step 2 — port 3-4 cheap architectural tweaks.**
Untied embedding/head, QK-norm, ReLU² MLP, logit softcap=15. All are 1-5 line changes to the model definition.

**Step 3 — warmup the attention window.**
Our chunked decoder naturally has a short context per chunk. Warmup from "intra-chunk only" to "inter-chunk attention allowed over last 2 chunks" to "full context" over the first 30% of training steps.

**Step 4 — add gated value-skip.**
A gated skip from compressor output to chunk-decoder output is the architectural heir of record 55. Promising for our hierarchical structure.

**Step 5 — measure against BPC, not loss.**
The speedrun gates against val cross-entropy nats; we gate against `BPC = val_loss / ln(2)`. Log both through W&B. The `ml-training-loop/SKILL.md` already documents the BPC logging patch in `WBMetricRenamer.on_epoch_end`.

## Provenance

This document distills:
- `README.md` from KellerJordan/modded-nanogpt (5.6k stars, 89 records through 2026-07).
- `train_gpt.py`, `train_gpt_medium.py` (current record code).
- `triton_kernels.py`, `dc_triton_kernels.py` (FP8/custom kernels, not applicable to TPU).
- The record-by-record history in the README's record table.

License: MIT. Upstream: https://github.com/KellerJordan/modded-nanogpt. Pull requests there are competitive benchmarks. Nothing here is copied verbatim; it is a distillation oriented toward our TPU/PerPos setting.
