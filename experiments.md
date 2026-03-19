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
