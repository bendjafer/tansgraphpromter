# GraphPrompter — Paper Summary

> "Can we Soft Prompt LLMs for Graph Learning Tasks?"  
> Liu, He, Tian, Chawla — WWW '24 Companion (arXiv:2402.10359)

---

## Core idea

GraphPrompter treats the GNN output as a **soft prompt** for a frozen LLM.
Instead of converting graph structure to text (which loses structural information),
it encodes a node's local subgraph with a GNN, projects the resulting embedding
into the LLM's token space, and prepends it to the node's text embeddings before
the LLM's self-attention layers. The LLM then generates the class label as free text.

This avoids the main weakness of prior work: mapping graph → text discards
neighbourhood topology. A GNN is better at aggregating structural information;
an LLM is better at reasoning about text. GraphPrompter lets each do what it
does best.

---

## Architecture (two sections)

### Graph section
```
target node v_i
    → 3-hop subgraph G_{s_i}  (all nodes within 3 hops)
    → GNN (GAT, 4 layers)
    → node embedding  X_i ∈ ℝ^{d_g}          (captures topology)
    → MLP projector   X̂_i = MLP(X_i) ∈ ℝ^{d_l}  (aligns to LLM dim)
```

### LLM section
```
node text T_i  (title, abstract, etc.)
    → frozen LLM tokenizer
    → T_tokens  (discrete token ids)
    → frozen LLM embedding table
    → T_emb ∈ ℝ^{M × d_l}
```

### Combined input to LLM
```
[BOS] [X̂_i]  [T_emb]  [question prompt]
       ↑           ↑
  graph soft    text tokens
  prompt token  (frozen embeddings)
       └─────── concatenated → LLM self-attention → generates label as text
```

The graph embedding occupies **one extra token position** prepended before the text.
The LLM is autoregressive and generates the class name as a natural-language string.

---

## What trains vs what's frozen

| Component | Default (`llm_frozen True`) | With LoRA (`llm_frozen False`) |
|-----------|---------------------------|-------------------------------|
| GNN (GAT/GCN, 4 layers) | ✅ trained | ✅ trained |
| MLP projector (2 layers) | ✅ trained | ✅ trained |
| LLM backbone (LLaMA-2-7B) | ❄️ frozen | LoRA on q,v projections (r=8, α=16) |

With the frozen LLM only ~8M parameters are updated (GNN + projector).
GraphPrompter + LoRA is the best-performing variant.

---

## Inputs the model requires

For each node `v_i` at training/inference time:

| Input | What it is | Where it comes from |
|-------|-----------|---------------------|
| Graph `G = (V, E)` | Adjacency structure + node features | PyG Data object |
| Node features `x` | Initial node feature matrix (e.g. bag-of-words, BERT embeddings) | Stored in `graph.x` |
| Node text `T_i` | Free-form text string per node | `dataset[i]['desc']` field |
| Question prompt | Classification question listing the class names | `dataset[i]['question']` field (shared across all nodes) |
| Label `y_i` | Ground truth class name as text | `dataset[i]['label']` field (training only) |

**The node text `desc` is mandatory** — it is tokenized and embedded by the LLM's
own embedding table and concatenated with the graph soft prompt. Without it,
the LLM has nothing to reason about semantically.

### How node text enters the code

In `src/model/graph_llm.py → forward()`:
```python
questions   = self.tokenizer(samples["question"], ...)
descriptions = self.tokenizer(samples["desc"], ...)   # ← node text here
labels      = self.tokenizer(samples["label"], ...)
graph_embeds = self.encode_graphs(samples)            # ← GNN soft prompt

# assembled as: [BOS] [graph_token] [desc tokens] [question tokens] [label tokens]
input_ids = descriptions.input_ids[i][:max_txt_len] + questions.input_ids[i] + label_ids
inputs_embeds = cat([bos, graph_embeds[i], word_embedding(input_ids)])
```

The `desc` field is truncated to `--max_txt_len` tokens (default 512) to fit GPU memory.

---

## Datasets in the paper (all are Text-Attributed Graphs)

| Dataset | Nodes | Edges | Classes | Node text |
|---------|-------|-------|---------|-----------|
| Cora | 2,708 | 5,429 | 7 | Paper title + abstract |
| Citeseer | 3,327 | 4,732 | 6 | Paper title + abstract |
| PubMed | 19,717 | 44,338 | 3 | Paper title + abstract |
| Ogbn-arxiv | 169,343 | 1,166,243 | 40 | Paper title + abstract |
| Ogbn-products | ~316,000 | 61M | 47 | Product title + description |

**All five datasets already have node text.** GraphPrompter was designed for TAGs
and has not been tested on text-free graphs.

---

## Key results

Node classification accuracy (mean over 5 seeds, LLaMA-2-7B backbone):

| Method | Cora | Citeseer | PubMed | Ogbn-arxiv | Ogbn-products |
|--------|------|----------|--------|------------|---------------|
| GAT (pure GNN) | 84.69 | 70.78 | 84.09 | 71.82 | 79.52 |
| Zero-Shot LLM | 43.31 | 29.22 | 91.39 | 44.23 | 15.05 |
| Soft Prompt Tuning | 70.31 | 70.97 | 91.45 | 71.99 | 75.14 |
| Fine-tuning + LoRA | 75.97 | 73.45 | 94.68 | 74.58 | 78.99 |
| **GraphPrompter + LoRA** | **80.26** | **73.61** | **94.80** | **75.61** | **79.54** |

GraphPrompter + LoRA is best or runner-up on all datasets.

---

## Integration notes (relevant to connecting with TANS)

1. **The `desc` field is the only text input per node.** It is a plain Python string.
   Any source of node text — including LLM-generated descriptions from TANS —
   can be dropped in here.

2. **Node features `graph.x` are separate from `desc`.** The GNN uses `graph.x`
   (e.g. bag-of-words features) for its soft prompt; the LLM uses `desc` (raw text).
   For TANS-attributed graphs, `graph.x` may need to be synthesized
   (e.g. from the generated text, or set to a constant/identity).

3. **The dataset class controls how `desc` is built.** Each dataset class
   (e.g. `CiteseerDataset.__getitem__`) formats the desc string:
   ```python
   'desc': f'Abstract: {text["abstract"]}'
   ```
   To plug in TANS text, this is the field to replace.

4. **The graph structure `(V, E)` is unchanged.** TANS generates text from topology
   but doesn't modify the graph itself — the same edges and node IDs carry over.

5. **No changes to the GNN, projector, or LLM are needed.** The integration point
   is purely in how `desc` is populated for each node.