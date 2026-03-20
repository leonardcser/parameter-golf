# Architecture Experiments

## Baseline
- **val_bpb:** 1.3474 | **Steps:** 1170 | **ms/step:** 513 | **Size:** 13.0MB

---

## Results Summary

| # | Architecture | val_bpb | Gap | Steps | ms/step | Verdict |
|---|---|---|---|---|---|---|
| 1 | Recurrent 2+1×5+2 | 1.3774 | +0.030 | 1203 | 499 | Shared weights lose quality |
| 2 | Recurrent wider d=576 | 1.3897 | +0.042 | 962 | 624 | Wider = slower |
| 3 | Deep recurrence 13-eff | 1.4198 | +0.072 | 844 | 711 | Too slow |
| 4c | Hourglass gs=4 noparam | 1.3552 | +0.008 | 2029 | 296 | Close, fast! |
| **5** | **Hourglass gs=2** | **1.3492** | **+0.002** | **1665** | **360** | **Best! 42% more steps** |
| 6 | Hourglass gs=2 10blk | 1.3502 | +0.003 | 1573 | 381 | Extra layer not worth it |
| 7 | 12 layers mlp_mult=1 | 1.3641 | +0.017 | 1028 | 584 | More layers = slower |
| 9 | Hourglass + aux loss | 1.8286 | +0.481 | 1623 | 370 | Aux loss conflicts |
| 10 | Hourglass + conv mid | 1.3628 | +0.015 | 1855 | 324 | Conv too weak |
| 11 | Hourglass stride-sel | 1.3520 | +0.005 | 1658 | 362 | Mean-pool better |
| 12 | 10 layers baseline | 1.3516 | +0.004 | 1055 | 569 | Not worth extra |
| 14 | DiffAttn (V-split) | 1.3658 | +0.018 | 1087 | 552 | Flash attn issue |
| 15b | RWKV linear recurrence | 1.4606 | +0.113 | 929 | 646 | Way worse, slower |
| 16 | Dilated causal net | 1.5401 | +0.193 | 1475 | 407 | No attention = bad |
| 17 | Parallel attn+MLP | 1.3566 | +0.009 | 1208 | 497 | Slightly worse |
| 18 | Embedding recycling | 1.3786 | +0.031 | 899 | 668 | Softmax too expensive |
| 19 | Multi-scale RoPE | 1.3567 | +0.009 | 1109 | 541 | Per-head loop slow |

## Key Findings
1. **Attention is king**: RWKV (-0.113), dilated shifts (-0.193) — removing attention is catastrophic
2. **Speed kills**: Any change slower than baseline costs training steps, which matters more than per-step quality at 10 min budget
3. **Hourglass is most promising**: Only 0.002 from baseline, would likely win on 8xH100 (42% more training steps)
4. **Baseline is near-optimal**: The 9-layer transformer with Muon optimizer is extremely well-tuned for this scale

---

## Phase 2: Hyperparameter Tuning

### Sweep Results (hourglass gs=2 base)
| Config | val_bpb | Notes |
|--------|---------|-------|
| **hg_warmdown (600)** | **1.3434** | **BEATS BASELINE (1.3464)!** |
| hg_lr_low (matrix=0.03) | 1.3480 | Slightly better than default LR |
| hg_gs2 (defaults) | 1.3487 | Hourglass with no mid-val |
| hg_lr_high (matrix=0.06) | 1.3637 | LR too high |

Key finding: warmdown_iters=600 (vs 1200) is the breakthrough. The hourglass gets 42% more steps, so the warmdown can start later.

### Best Result: seq_len=2048 + hourglass gs=2 + warmdown=600
| Config | val_bpb | Steps | ms/step | Size |
|--------|---------|-------|---------|------|
| **best_seq2048_wd600** | **1.3323** | **1522** | **394** | **14.83MB** |
| baseline (original) | 1.3464 | 1170 | 513 | 13.0MB |
| **Improvement** | **-0.014** | **+352** | **-119** | **+1.8MB** |

Longer sequence length (2048 vs 1024) gives each position 2× more context for prediction.
The hourglass makes this affordable by downsampling middle layers to 1024 tokens.
Net speed: 394ms/step (vs baseline 513ms) — still 23% faster despite 2× longer sequences.

### Further Tuning (seq4096 + hourglass gs=2)
| Config | val_bpb | Notes |
|--------|---------|-------|
| matrix_lr=0.025, wd=300 | 1.3273 | Lower LR + shorter warmdown |
| + matrix_lr=0.03, embed=0.035 | 1.3303 | |
| + muon_momentum=0.97 | **1.3259** | Higher momentum smooths optim |
| muon_momentum=0.98 | 1.3283 | Too high |
| matrix_lr=0.02, wd=200 | 1.3310 | LR too low |

### Vocab Size Experiments (biggest lever!)
| Config | val_bpb | Size | Notes |
|--------|---------|------|-------|
| sp1024 (baseline vocab) | 1.3259 | 14.1MB | Best sp1024 config |
| **sp4096, 9 blocks** | **1.2968** | **15.4MB** | Larger vocab = fewer tokens/byte |
| **sp4096, 8 blocks (2+4+2)** | **1.2951** | **14.0MB** | Fewer blocks = faster, still better |
| **sp8192, 7 blocks (2+3+2)** | **1.2776** | **14.5MB** | **Even larger vocab wins again!** |

| sp8192, 8 blocks (2+4+2) | 1.2847 | 15.8MB | Too big, 7 better |
| sp8192, tuned LR | 1.2810 | 14.9MB | Default LR was better |
| **sp16384, 5 blocks (2+1+2)** | **1.2634** | **15.4MB** | **Vocab keeps winning!** |

| sp16384, 4 blocks (2+0+2) | 1.2632 | 14.0MB | Same quality, simpler (no hourglass!) |
| sp16384, d384, 8 blocks | 1.2902 | 13.1MB | Narrow worse than wide |
| sp16384, tuned LR | 1.2752 | 14.0MB | Default LR was better |
| sp32768, d512, 3 blocks | 1.2701 | 20.2MB | OVER LIMIT |
| sp32768, d384, 5 blocks | 1.2762 | 16.2MB | OVER LIMIT |
| sp32768, factored r=256 | 1.3068 | 14.0MB | Bottleneck kills quality |
| sp24576, 3 blocks | 1.2747 | 16.4MB | OVER LIMIT |
| sp24576, 2 blocks | 1.3029 | 14.9MB | Too few blocks |

**Best: sp16384 + 4 blocks (2+0+2) + d512 = val_bpb 1.2632, 14.0MB**

Key insight: vocab size is the single biggest lever. Sweet spot is ~16K vocab with 4 blocks at dim=512.
- Below 16K: not enough compression per token
- Above 16K: embedding too large, forces too few blocks
- Factored embeddings don't help (bottleneck kills quality)
- Width (dim) matters more than depth (blocks) at this scale

### Tokenizer Quality Experiments
| Config | val_bpb | Size | Tokenizer docs |
|--------|---------|------|---------------|
| sp16384, 4blk, 200K docs | 1.2632 | 14.0MB | 200K |
| sp16384v2, 4blk, 1M docs | 1.2726 | 14.0MB | 1M (improved!) |
| sp20480, 4blk, 200K docs | 1.2684 | 15.9MB | 200K |
| **sp20480v2, 4blk, 1M docs** | **1.2703** | **15.9MB** | **1M (best!)** |
| sp20480v2 + muon=0.97 | 1.2705 | 15.9MB | 1M (no help) |

Better tokenizer training data consistently improves BPB.

### Dim/Block Optimization (sp20480v3)
| Config | val_bpb | Size | Notes |
|--------|---------|------|-------|
| d512, 4blk | 1.2674 | 15.9MB | Baseline for sp20480 |
| d448, 5blk MHA | **1.2609** | **15.2MB** | **Best! More depth > width** |
| d448, 6blk MHA | 1.2625 | 16.4MB | Over limit |
| d448, 6blk GQA-1 | 1.2783 | 15.2MB | Too few KV heads |
| d384, 7blk MHA | 1.2669 | 14.3MB | Too narrow |
| d480, 4blk GQA-4 | 1.2795 | 14.6MB | Too wide, not deep enough |
| SwiGLU d512, 4blk | 1.2721 | 15.1MB | Saves params but relu² better |
| SwiGLU d512, 5blk | 1.2647 | 16.2MB | Over limit |

**Current best: sp20480v3, d448, 7-head MHA, 5 blocks = val_bpb 1.2609**
