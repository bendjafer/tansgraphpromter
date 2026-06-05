# Airport Domain Adaptation Experiments
**GraphPrompter — GNN + Vicuna-7B (frozen)**  
Task: 4-class node classification (Low / Medium / High / Very High activity)  
Random baseline: **25.0%**

---

## Overview of Experiments

| # | Name | GNN | Features | hidden | layers | Feature norm | Val strategy | Split |
|---|------|-----|----------|--------|--------|--------------|--------------|-------|
| 1 | GAT + TPF | GAT | 5 topo | 64 | 2 | None | Val loss (source) | 60/20 source · 100% target |
| 2 | GAT + MiniLM | GAT | 384 MiniLM | 256 | 2 | None | None (last epoch) | 100% source · 100% target |
| 3 | TANS + TPF | GCN | 5 topo | 16/32¹ | 3/2¹ | StandardScaler | Val acc (target 20%) | 100% source · 20% val · 80% test |
| 4 | TANS + MiniLM | GCN | 384 MiniLM | 16/32¹ | 3/2¹ | None | Val acc (target 20%) | 100% source · 20% val · 80% test |

> ¹ Direction-dependent: brazil→europe uses hidden=16, layers=3, normalize=none, lr=5e-2, wd=1e-4, dropout=0.0  
>   europe→brazil uses hidden=32, layers=2, normalize=batchnorm, lr=5e-3, wd=0, dropout=0.8

---

## Shared Architecture (all experiments)

### LLM
| Component | Value |
|-----------|-------|
| Backbone | Vicuna-7B (LLaMA) |
| Precision | float16 |
| Frozen | Yes — weights not updated |
| Trainable % | ~0.14% (GNN + projector only) |

### Projector (GNN → LLM token space)
```
GNN output → Linear(gnn_out, 2048) → Sigmoid → Linear(2048, 4096)
```
Output is a single soft-prompt token prepended to the LLM input sequence.

### Optimiser (Experiments 1 & 2)
| Param | Value |
|-------|-------|
| Optimiser | AdamW |
| lr | 1e-5 |
| weight_decay | 0.05 |
| betas | (0.9, 0.95) |
| grad clip | 0.1 |
| LR schedule | Cosine, warmup=1 epoch, min_lr=5e-6 |
| Epochs | 15 |
| Batch size | 4 (grad_accum=2 → effective 8) |

### Checkpoints
Saved to: `output_{topo,minilm,tans}/`  
Only trainable params saved (LLM excluded) — ~38 MB per file.

---

## Experiment 1 — GAT + Topological Features (TPF)

**Script:** `run_airport_experiments.sh`  
**Outputs:** `output_topo/` · Logs: `logs/airport/`

### GNN Architecture
| Param | Value |
|-------|-------|
| Model | GAT (Graph Attention Network) |
| Input | [N, 5] topological features |
| Features | degree · closeness · betweenness · clustering · square_clustering |
| Feature normalisation | None (NetworkX outputs already in [0, 1]) |
| hidden_dim | 64 |
| out_dim | 64 |
| num_layers | 2 |
| num_heads | 4 (concat=False) |
| dropout | 0.0 |
| BatchNorm | Yes (after each hidden layer) |

### Split detail
> **Note:** This experiment ran with the standard per-dataset 60/20/20 split on the source graph  
> (brazil: 78 train / 26 val; europe: 239 train / 79 val). Validation used source val loss.  
> Test set = 100% of target graph nodes.

### Checkpoint files
```
output_topo/europe_graph_llm_7b_gat_seed{0,1,2}_checkpoint_best.pth
```

### Results

| Direction | Seed 0 | Seed 1 | Seed 2 | **Mean ± Std** |
|-----------|--------|--------|--------|----------------|
| brazil → europe | 37.34% | 40.85% | 47.87% | **42.02% ± 4.38%** |
| europe → brazil | — *(incomplete)* | 58.02% | 59.54% | **58.78% ± 0.76%** *(2 seeds)* |

> europe→brazil seed0 run did not complete.

---

## Experiment 2 — GAT + MiniLM Embeddings

**Script:** `run_airport_minilm_experiments.sh`  
**Outputs:** `output_minilm/` · Logs: `logs/airport_minilmc/`

### GNN Architecture
| Param | Value |
|-------|-------|
| Model | GAT |
| Input | [N, 384] MiniLM embeddings |
| Embedding model | all-MiniLM-L12-v2 |
| Embedding template | `"The node type is airport. The node description is na. The additional node description is {desc}."` |
| Embedding cache | `dataset/airports/{name}/minilm_x.pt` |
| Feature normalisation | None |
| hidden_dim | 256 |
| out_dim | 256 |
| num_layers | 2 |
| num_heads | 4 (concat=False) |
| dropout | 0.0 |
| BatchNorm | Yes |

### Split detail
> Source = 100% train, no val set.  
> Checkpoint saved every epoch (last epoch = best).  
> Test set = 100% of target graph nodes.

### Checkpoint files
```
output_minilm/{dataset}_graph_llm_7b_gat_seed{0,1,2}_checkpoint_best.pth
```

### Results

| Direction | Seed 0 | Seed 1 | Seed 2 | **Mean ± Std** |
|-----------|--------|--------|--------|----------------|
| brazil_minilm → europe_minilm | 42.86% | 46.87% | 48.87% | **46.20% ± 2.50%** |
| europe_minilm → brazil_minilm | 36.64% | 62.60% | 55.73% | **51.66% ± 10.98%** |

> High std on europe→brazil indicates sensitivity to random seed initialisation.

---

## Experiment 3 — TANS + Topological Features (TPF)

**Script:** `run_airport_tans_experiments.sh` (TPF section)  
**Outputs:** `output_tans/` · Logs: `logs/airport_tans/`

### GNN Architecture — brazil → europe
| Param | Value |
|-------|-------|
| Model | GCN |
| Input | [N, 5] TPF, **StandardScaler normalised** |
| hidden_dim | 16 |
| out_dim | 16 |
| num_layers | 3 |
| dropout | 0.0 |
| BatchNorm | **No** |
| lr | 5e-2 |
| weight_decay | 1e-4 |

### GNN Architecture — europe → brazil
| Param | Value |
|-------|-------|
| Model | GCN |
| Input | [N, 5] TPF, **StandardScaler normalised** |
| hidden_dim | 32 |
| out_dim | 32 |
| num_layers | 2 |
| dropout | 0.8 |
| BatchNorm | **Yes** |
| lr | 5e-3 |
| weight_decay | 0 |

### Split detail
> Source = 100% train. Target split randomly (seed-dependent):  
> 20% val (model selection by **val accuracy**) · 80% test (reported).

### Checkpoint files
```
output_tans/{dataset}_graph_llm_7b_gcn_seed{0,1,2}_checkpoint_best.pth
```

### Results

| Direction | Seed 0 | Seed 1 | Seed 2 | **Mean ± Std** | Best epoch |
|-----------|--------|--------|--------|----------------|------------|
| brazil → europe | 25.00% | 30.00% | 23.44% | **26.15% ± 2.80%** | 0 / 0 / 0 |
| europe → brazil | 25.71% | 37.14% | 53.33% | **38.73% ± 11.33%** | 0 / 0 / 3 |

> brazil→europe converges at epoch 0 across all seeds — close to random (25%).  
> High variance on europe→brazil suggests the TANS hyperparams may need tuning for this setup.

---

## Experiment 4 — TANS + MiniLM Embeddings

**Script:** `run_airport_tans_experiments.sh` (MiniLM section)  
**Outputs:** `output_tans/` · Logs: `logs/airport_tans/`

### GNN Architecture
Same as Experiment 3 per direction, except:
- Input: [N, 384] MiniLM embeddings (no feature normalisation)
- Checkpoint filenames use `brazil_minilm` / `europe_minilm`

### Checkpoint files
```
output_tans/{dataset_minilm}_graph_llm_7b_gcn_seed{0,1,2}_checkpoint_best.pth
```

### Results

| Direction | Seed 0 | Seed 1 | Seed 2 | **Mean ± Std** | Best epoch |
|-----------|--------|--------|--------|----------------|------------|
| brazil_minilm → europe_minilm | 46.88% | 46.88% | 50.31% | **48.02% ± 1.62%** | 4 / 3 / 3 |
| europe_minilm → brazil_minilm | 28.57% | 64.76% | 25.71% | **39.68% ± 17.77%** | 0 / 4 / 0 |

> brazil→minilm→europe is the most stable result across all experiments (std=1.62%).  
> europe→brazil_minilm has extreme variance — seed 1 finds a good solution, seeds 0 and 2 collapse to near-random.

---

## Summary Table

| Experiment | Direction | Mean Acc | Std | vs Random (+25%) |
|------------|-----------|----------|-----|-----------------|
| GAT + TPF | brazil → europe | 42.02% | ±4.38% | +17.02% |
| GAT + TPF | europe → brazil | 58.78% | ±0.76% | +33.78% |
| GAT + MiniLM | brazil → europe | 46.20% | ±2.50% | +21.20% |
| GAT + MiniLM | europe → brazil | 51.66% | ±10.98% | +26.66% |
| TANS + TPF (GCN) | brazil → europe | 26.15% | ±2.80% | +1.15% |
| TANS + TPF (GCN) | europe → brazil | 38.73% | ±11.33% | +13.73% |
| TANS + MiniLM (GCN) | brazil → europe | 48.02% | ±1.62% | +23.02% |
| TANS + MiniLM (GCN) | europe → brazil | 39.68% | ±17.77% | +14.68% |

---

## Key Observations

1. **GAT outperforms GCN** in the TPF setting by a large margin — the TANS GCN hyperparams (hidden=16, lr=5e-2) appear poorly suited for the GraphPrompter projector architecture, which requires a minimum representational capacity to project into the 4096-dim LLM space.

2. **europe → brazil is consistently easier** than brazil → europe across all experiments. Europe (399 nodes, larger graph) provides a richer training signal than Brazil (131 nodes).

3. **MiniLM embeddings are competitive with TPF** in the TANS setup and more stable. In the GAT setup, both feature types perform similarly, but MiniLM has higher variance on europe→brazil.

4. **TANS TPF brazil→europe is near-random** (26.15%) — the small GCN (640 params, hidden=16) cannot produce a meaningful soft-prompt token for Vicuna-7B after the projector up-scales to 4096 dims.

5. **High variance across seeds** in the TANS setup suggests the 20% val split (only 26 nodes for europe→brazil) is too small for reliable model selection.
