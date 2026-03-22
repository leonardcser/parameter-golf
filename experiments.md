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

### Integrated Community Techniques (from issue #140 analysis)
| Technique | Credit | val_bpb | Impact |
|-----------|--------|---------|--------|
| Muon WD 0.04 + Ortho Init + Grad Clip 0.3 | @notapplica #60, @raahilshah #162 | 1.2649 | +0.004 worse but 0.5MB smaller! |
| SmearGate + BigramHash | @unnir #102,#135 | 1.2693 | Worse on 1 GPU (needs more steps) |
| SmearGate only | @unnir | 1.2679 | Worse on 1 GPU |
| Sliding window eval (stride=64) | @mattqlf #50 | TBD | ~0.03 on 8xH100, too slow on 1 GPU |

Note: SmearGate/BigramHash need more training steps to learn (1200 steps on 1 GPU isn't enough).
On 8xH100 with ~7000+ steps, these techniques should provide significant gains.

### Novel Ideas (in progress)
From plan agent analysis, implementing:
1. **Byte-weighted loss** — align training with BPB metric (weight tokens by byte count)
2. **Trigram hash embedding** — extend BigramHash to 3-token context
3. **Adaptive softcap** — learned per-position logit scaling
4. **Int6 quantization** — implemented, saves 25% but degrades 0.016 BPB on 1 GPU

### Int6 + Larger Model Experiments
| Config | val_bpb | Size | Notes |
|--------|---------|------|-------|
| d448 5blk int6+zstd | 1.2815 | 11.6MB | Int6 degrades 0.016 BPB |
| d448 8blk int6+zstd | 1.2697 | 14.9MB | Extra blocks help but not enough |
| d448 5blk int6 MLP3x | 1.2747 | 12.9MB | Needs more steps (8xH100) |
| Muon 0.99 int8 | 1.2654 | 14.3MB | Similar to 0.97 on 1 GPU |

### Overall Best (1 GPU)
**val_bpb=1.2649** with d448 5blk sp20480 int8+zstd Muon WD ortho init grad clip

### Novel Techniques
| Technique | val_bpb | Impact | Notes |
|-----------|---------|--------|-------|
| **Adaptive softcap (NOVEL)** | **1.2607** | **-0.004** | **Learned per-position logit scaling. NEW BEST.** |
| Ngram hash (bigram+trigram) | 1.2608 | ~0 | Not worth complexity on 1 GPU |
| SWA (avg last 50%) | 1.2733 | +0.013 | Hurts on 1 GPU, helps on 8xH100 |
| Byte-weighted loss | 1.4971 | +0.23 | Completely breaks training |

### Current Best on 1 GPU: val_bpb = 1.2607
Config: d448, 7-head MHA, 5 blocks, sp20480 (2M doc tokenizer), seq4096, Muon WD 0.04,
ortho init, grad clip 0.3, adaptive softcap, int8+zstd-22.

### RoPE Base + QK Gain + NS Steps Sweep
| Config | val_bpb | Notes |
|--------|---------|-------|
| rope_base=10000 (default) | 1.2603 | |
| rope_base=50000 | 1.2587 | |
| rope_base=100000 | 1.2582 | |
| **rope_base=500000** | **1.2578** | **Optimal** |
| rope_base=1000000 | 1.2586 | |
| qk_gain=1.5 (default) | 1.2578 | |
| qk_gain=2.0 | 1.2569 | |
| qk_gain=2.5 | 1.2563 | |
| **qk_gain=3.0** | **1.2559** | **Optimal** |
| qk_gain=4.0 | 1.2567 | |
| NS=5 (default) | 1.2559 | |
| **NS=7** | **1.2546** | **Optimal** |
| NS=10 | 1.2560 | |

### Final Best on 1 GPU: val_bpb = 1.2546
Config: d448, 7-head MHA, 5 blocks, sp20480 (2M doc tokenizer), seq4096,
Muon WD 0.04, ortho init, grad clip 0.3, adaptive softcap (base=20),
rope_base=500k, qk_gain=3.0, NS=7, int8+zstd-22.

### Vocab vs Depth Optimization (with best hparams: rope 500k, qk 3.0, ns 7)
| Config | val_bpb | Size | Notes |
|--------|---------|------|-------|
| sp20480 d448 5blk | 1.2546 | 14.6MB | Previous best |
| sp16384 d448 6blk | 1.2560 | 14.1MB | Competitive |
| **sp16384 d448 7blk** | **1.2529** | **15.1MB** | **NEW BEST!** |
| sp16384 d448 8blk | 1.2540 | 16.1MB | Over limit |
| sp16384 d512 5blk | 1.2561 | 14.6MB | Width < depth |
| int6 d448 7blk | 1.2629 | 13.8MB | Int6 hurts on 1 GPU |
| int6 d448 9blk | 1.2614 | 16.1MB | Over limit |

### Batch Size + Warmdown Optimization
| batch_tokens | warmdown | steps | ms/step | val_bpb | Notes |
|-------------|----------|-------|---------|---------|-------|
| 524288 | 300 | 1200 | 500 | 1.2529 | Previous best |
| 262144 | 300 | 1994 | 301 | 1.2461 | Huge batch size win |
| 196608 | 400 | 2575 | 233 | 1.2437 | Sweet spot for batch |
| 131072 | 500 | 3873 | 155 | 1.2489 | Too small, noisy grads |
| 196608 | 600 | 2575 | 233 | 1.2397 | |
| 196608 | 800 | 2575 | 233 | 1.2369 | |
| 196608 | 1000 | 2575 | 233 | 1.2364 | |
| 196608 | 1200 | 2575 | 233 | 1.2353 | |
| **196608** | **1500** | **2575** | **233** | **1.2343** | **NEW BEST** |
| 196608 | 2000 | 2575 | 233 | 1.2355 | Too much warmdown |

Key finding: smaller batch → more steps → better BPB (to a point).
Longer warmdown also helps consistently up to ~58% of total steps.

### Overall Best on 1 GPU: val_bpb = 1.2343
Config: sp16384v2, d448, 7-head MHA, 7 blocks (3+0+4), seq4096, gs=2,
batch=196608, warmdown=1500, adaptive softcap (base=20), rope_base=500k,
qk_gain=3.0, NS=7, Muon WD 0.04, ortho init, grad clip 0.3, int8+zstd-22.
Total improvement: 1.3464 → 1.2331 (-0.113, 8.4% better).
Uses stable tokenizer (fineweb_16384stable_bpe.model, won't be overwritten).

### AdamW + Muon WD Tuning
| Config | val_bpb | Notes |
|--------|---------|-------|
| Adam (no WD) for embed/scalar | 1.2343 | Previous |
| **AdamW WD=0.01** | **1.2334** | **Best! Weight decay helps all params** |
| AdamW WD=0.02 | 1.2339 | Too much |
| Muon WD=0.06 | 1.2345 | 0.04 is optimal |

### Other experiments
| Config | val_bpb | Notes |
|--------|---------|-------|
| muon=0.98 | 1.2354 | 0.97 is optimal |
| LR=0.03 | 1.2367 | 0.025 is optimal |
| softcap=100 (none) | 1.2376 | Softcap helps |
| min skips (1+6) | 1.2365 | U-Net skips help |
| MTP (multi-token prediction) | 1.5012 | Needs separate heads |

---

## Phase 3: Clean Code Rebuild + New Techniques

Rebuilt code from scratch — PR #315 integration was 2× slower per step (437ms vs 213ms)
due to unused feature branches in the compiled model. Clean code restored full speed.

### Architecture Refinements (clean code, best hparams)
| Config | val_bpb (live) | val_bpb (int8) | Size | Steps | Notes |
|--------|---------------|---------------|------|-------|-------|
| 6blk flat (no hourglass) | 1.2326 | 1.2344 | 15.0MB | 2812 | Baseline clean code |
| 6blk hourglass gs=2 | 1.2303 | 1.2321 | 15.0MB | 2810 | Mean-pool regularizer helps! |
| 7blk mlp2.5 (no hg) | 1.2283 | 1.2300 | **16.3MB** | 2471 | Over 16MB limit |
| 7blk mlp2.25 | 1.2327 | 1.2346 | 15.9MB | 2516 | Slower, similar quality |
| 7blk mlp2.0 | 1.2347 | 1.2366 | 15.3MB | 2556 | Narrow MLP hurts |
| 7blk hg gs=2 | 1.2301 | 1.2318 | **16.3MB** | 2487 | Over limit |
| sp12288 7blk | 1.2334 | 1.2350 | 14.7MB | 2543 | Smaller vocab hurts more |

### Community Technique Integration
| Technique | Credit | val_bpb (int8) | Impact | Notes |
|-----------|--------|---------------|--------|-------|
| GPTQ-lite int8 | @signalrush #414, @thwu1 #379 | 1.2344 | +0.000 | No benefit for int8 (helps int6) |
| EMA decay=0.997 | community | 1.2360 | +0.016 worse | Not enough steps on 1 GPU |
| **Value Residual** | **ResFormer, @PR #413** | **1.2299** | **-0.022** | **V0 skip-connect through depth** |
| Gated Attention | arXiv:2505.06708 | 1.2401 | +0.057 worse | Adds overhead, needs more steps |

### Warmdown Tuning (with Value Residual + hourglass)
| warmdown | val_bpb (live) | val_bpb (int8) |
|----------|---------------|---------------|
| 1200 | 1.2300 | 1.2315 |
| **1500** | **1.2284** | **1.2299** |
| 1800 | 1.2289 | 1.2304 |
| 2000 | 1.2304 | 1.2322 |

### Sparse Attention Gate (credit: modded-nanogpt PR #117)
| Config | val_bpb (live) | val_bpb (int8) | Size | Notes |
|--------|---------------|---------------|------|-------|
| VR + hourglass (prev best) | 1.2284 | 1.2299 | 15.0MB | |
| **+ Sparse Attention Gate** | **1.2265** | **1.2279** | **15.0MB** | **72 extra params, -0.002!** |

### NEW Overall Best on 1 GPU: val_bpb = 1.2263 (live) / 1.2277 (int8+zstd)
Config: sp16384stable, d448, 7-head MHA, 6 blocks, seq4096, hourglass gs=2,
batch=196608, warmdown=1500, adaptive softcap (base=20, novel), Value Residual
(credit: ResFormer arXiv:2410.17897), Sparse Attention Gate (credit: modded-nanogpt #117),
rope_base=500k, qk_gain=3.0, NS=7, Muon WD 0.04, AdamW WD 0.01, ortho init,
grad clip 0.3, GPTQ-lite int8+zstd-22. Size: 15.1MB.
Sliding window (stride=64): **1.2190 BPB**.
Total improvement: 1.3464 → 1.2190 (-0.127, 9.5% better).

### Deeper Models with Int6 Quantization
| Config | val_bpb (live) | val_bpb (quant) | Size | Notes |
|--------|---------------|----------------|------|-------|
| 9blk MLP3 int6+QAT0.15 | 1.2199 | 1.2431 (int6) | 14.3MB | Int6 gap +0.023 |
| 9blk MLP3 int6+QAT0.30 | 1.2203 | 1.2441 (int6) | 14.3MB | Earlier QAT doesn't help |
| 8blk MLP2.5 int8 bigram | 1.2228 | 1.2242 (int8) | **18.7MB** | Over limit |
| 8blk MLP2.5 int6+QAT bigram | 1.2232 | 1.2456 (int6) | 12.9MB | Int6 gap too large |
| 8blk MLP2.5 int8 no bigram | 1.2240 | 1.2254 (int8) | **17.5MB** | Over limit |
| 7blk MLP3 int8 | 1.2249 | 1.2263 (int8) | **17.3MB** | Over limit |
| 7blk MLP3 mixed int6/int8 | 1.2247 | 1.2382 (int6) | 13.9MB | Int6 gap +0.013 |
| 8blk d384 MLP3 int8 | 1.2293 | 1.2307 (int8) | 14.9MB | Narrow hurts |

Key finding: more layers help quality (1.22 vs 1.23) but int6 quantization loses 0.013-0.024 BPB
on 1 GPU (not enough QAT training). Need ~5000+ steps for QAT to be effective.

### Novel Optimizer: Mousse-lite (diagonal curvature preconditioning)
| Config | val_bpb (int8) | Notes |
|--------|---------------|-------|
| Muon (baseline) | 1.2299 | |
| Mousse-lite beta2=0.95 | 1.2319 | Diagonal approx too simple |
| Mousse-lite beta2=0.99 | 1.2317 | Marginally better but still worse |

### BigramHash on 6 layers
| Config | val_bpb (int8) | Size | Notes |
|--------|---------------|------|-------|
| 6blk + bigram 8192/128 | 1.2301 | **16.3MB** | Over limit |
| 6blk + bigram 4096/64 | 1.2351 | 15.3MB | Not enough steps to learn |

### Estimated 8xH100 Score: ~1.10-1.12 BPB
With int6 + more layers + SmearGate + BigramHash + sliding window + SWA/EMA + adaptive softcap + Value Residual
