# GraphPrompter — Code Audit Report

---

## Project Overview

- **Paper**: "Can we Soft Prompt LLMs for Graph Learning Tasks?" (WWW '24, arXiv:2402.10359)
- **Task**: Node classification on Text-Attributed Graphs (TAGs)
- **Core idea**: Encode a node's subgraph with a GNN → project embedding into LLM token space → prepend as a single soft-prompt token before the text tokens; LLM generates the class label as free text
- **Backbone**: Vicuna-7B (LLaMA-2-7B) by default; T5 / original LLaMA adapter variants also present
- **Datasets**: Cora, Citeseer, PubMed, ogbn-arxiv, ogbn-products
- **Extra experiment**: `brazil/` subdirectory — generates TANS-style structural descriptions for the Brazil Airport graph using OpenAI API (unrelated to the main pipeline)

---

## File-by-File Breakdown

---

### `train.py`
- **Role**: Primary single-GPU training entry point for LLM-based models
- `main(args)` — full train/val/test loop with early stopping, saves best checkpoint
- Imports: `torch`, `wandb`, `peft`, `src.model`, `src.dataset`, `src.utils.*`
- Connects to: all `src/` modules
- ⚠️ `best_epoch` is referenced at line 107 (`Best Epoch {best_epoch}`) before it is ever assigned — crashes on first epoch if `val_loss` never improves (variable lives inside `if val_loss < best_val_loss:` block)
- Early stopping implemented via `args.patience`

---

### `train_ddp.py`
- **Role**: Multi-GPU (DDP) training entry point; uses timm optimizer factory + AMP scaler + TensorBoard
- `train_one_epoch(model, data_loader, optimizer, device, epoch, loss_scaler, ...)` — one training epoch with gradient accumulation
- `val_one_epoch(...)` — validation epoch; mirrors train but `no_grad`
- `test_one_epoch(...)` — only prints predictions, computes no metrics
- `main(args)` — DDP init, model wrapping, epoch loop
- Imports: `torch`, `wandb`, `timm`, `tensorboard`, `src.*`
- ⚠️ `data_loader_test` reuses `sampler_val` (the validation sampler) — test data and val data use the same sampler, so test may overlap val
- ⚠️ `test_one_epoch` computes no accuracy; only prints raw preds — no quantitative test result
- ⚠️ `model = load_model[...](graph=..., graph_type=..., args=args)` — omits the `prompt` kwarg that `GraphLLM.__init__` expects
- Checkpoint save frequency: every 8 epochs or final epoch (no best-model selection)

---

### `train_gnn.py`
- **Role**: Standalone GNN-only training entry point (no LLM); CrossEntropyLoss, val accuracy for model selection
- `main(args)` — trains GCN/GAT standalone on subgraph batches
- Imports: `torch`, `wandb`, `src.model`, `src.dataset`, `src.utils.*`
- ⚠️ `adjust_learning_rate` called with 4 args (`param_group, args.lr, epoch_frac, args`) but the function signature is 3 args (`param_group, epoch, args`) — `args.lr` is passed as `epoch`, `epoch_frac` as `args`. LR schedule is silently broken
- ⚠️ `best_epoch` referenced at line 121 before first assignment (same bug as `train.py`)

---

### `inference.py`
- **Role**: Inference-only entry point; loads the test split, runs `model.inference()`, computes accuracy
- `main(args)` — skips training; relies on a pre-trained checkpoint being present
- ⚠️ Does not call `_reload_best_model`; no checkpoint is actually loaded — model weights are random unless a checkpoint is provided via `--resume` (which `inference.py` never reads)

---

### `run.sh`
- **Role**: Experiment recipe / documentation of all paper commands
- Lists commands for: dense w/ instructions, dense w/o instructions, sparse semantics across datasets and model variants
- References git branches (`wo_question`, `sparse_semantics`) that may not exist on `main`

---

### `src/config.py`
- **Role**: All CLI arguments via `argparse`
- `parse_args_llama()` — returns flat namespace with model, dataset, LR, GNN, LLM, DDP, and output params
- ⚠️ `--llm_frozen` is a `str` (`'True'`/`'False'`), compared with `==` strings in model code — fragile (passing `--llm_frozen true` or `--llm_frozen 0` silently keeps the wrong behavior)
- ⚠️ `--patience` is `float` type but used as an integer epoch counter comparison

---

### `src/model/__init__.py`
- **Role**: Model registry (`load_model` dict) and LLM path registry (`llama_model_path`)
- `load_model` keys: `'graph_llm'`, `'llm'`, `'inference_llm'`, `'pt_llm'`, `'gcn'`, `'gat'`, `'llama_adapter'`, `'t5'`
- ⚠️ `llama_model_path` entries for `'13b'`, `'7b_chat'`, `'13b_chat'` are placeholder strings `'[Your LLM PATH]'` — will silently load garbage paths if selected

---

### `src/model/graph_llm.py` — `GraphLLM`
- **Role**: Core model; GNN encodes subgraph → MLP projects to LLM dim → prepended as soft token to text embeddings
- `__init__(graph, graph_type, prompt, args)` — loads LLM (Vicuna-7B via HuggingFace), builds GNN encoder + 2-layer MLP projector
- `encode_graphs(samples)` — runs GNN on subgraph, applies projector to target node(s); returns `[batch, 4096]`
- `forward(samples)` — assembles `[BOS][graph_token][desc_tokens][question_tokens][label_tokens]`, returns LM loss
- `inference(samples)` — same assembly minus labels, calls `model.generate()`, decodes predictions
- `print_trainable_params()` — counts trainable vs total params
- Imports: `torch`, `transformers` (AutoModelForCausalLM), `peft` (LoRA), `src.model.gnn`
- MLP projector: `Linear(gnn_out_dim→2048) → Sigmoid → Linear(2048→4096)` — hard-coded output dim 4096 (LLaMA hidden size)
- LoRA config: r=8, alpha=16, target `q_proj`/`v_proj`, dropout=0.05 — hard-coded, not exposed as args
- ⚠️ Max GPU memory hard-coded to `{0: "22GiB"}` — fails on smaller GPUs or multi-GPU setups
- ⚠️ Projector uses `Sigmoid` activation between two linears — unusual; ReLU/GELU is standard for alignment projectors

---

### `src/model/llm.py` — `LLM`
- **Role**: Text-only LLM baseline (no graph); optionally fine-tuned with LoRA
- `__init__(graph, graph_type, prompt, args)` — loads LLM; `graph`/`graph_type`/`prompt` params accepted but ignored
- `forward(samples)` / `inference(samples)` — same embedding assembly as `GraphLLM` but without the graph token
- LoRA config identical to `GraphLLM` (also hard-coded)
- ⚠️ Max memory set to `{0: '20GiB', 1: '20GiB', 2: '20GiB', 3: '20GiB'}` — assumes 4 GPUs

---

### `src/model/pt_llm.py` — `PromptTuningLLM`
- **Role**: Soft prompt tuning baseline; learns `num_virtual_tokens` trainable embeddings prepended to each sequence
- `__init__(prompt, args, **kwargs)` — loads frozen LLM, initializes `self.prompt` as a trainable parameter initialized from the instruction text embeddings
- `forward(samples)` / `inference(samples)` — prepends `prompt_embeds` (repeated per batch) then text embeddings
- `encode_graphs(ids)` — references `self.graph_encoder` and `self.graph` which are **never defined** — dead code, would crash if called
- ⚠️ `**kwargs` parameter in `__init__` is immediately shadowed by `kwargs = {"max_memory": ...}` on line 27 — the `graph` and `graph_type` passed by `train.py` are silently discarded
- ⚠️ 4-GPU memory assumption (`{0:'20GiB', 1:'20GiB', 2:'20GiB', 3:'20GiB'}`)

---

### `src/model/llama_adapter.py` — `LlamaAdapter`
- **Role**: Original LLaMA-Adapter (pre-HuggingFace); loads weights directly from `.pth` checkpoint
- `__init__(graph, graph_type, args)` — loads from hardcoded `'[Your LLM PATH]'`, freezes all except adapter and gate layers
- `forward(samples)` — uses custom `Tokenizer` (sentencepiece), constructs embeddings, passes to custom `Transformer`
- `inference(samples)` — manual autoregressive decoding (token-by-token), applies `sample_top_p`
- `sample_top_p(probs, p)` — top-p nucleus sampling
- ⚠️ LLM path is hardcoded as `'[Your LLM PATH]'` (line 25) — cannot be configured via CLI
- ⚠️ `ignore_index = 0` (not `-100`); token 0 is typically a valid pad/BOS token, not truly ignored by CrossEntropy
- ⚠️ `attention_mask` is commented out in `forward()` — may cause incorrect attention on padded inputs

---

### `src/model/t5.py` — `T5`
- **Role**: FLAN-T5-XL encoder-decoder baseline
- `__init__(graph, graph_type, args)` — loads `google/flan-t5-xl` (no graph use; `graph`/`graph_type` ignored)
- `forward(samples)` / `inference(samples)` — same pattern: embed desc+question as encoder input, label as decoder target
- ⚠️ All params trainable (no freeze logic), yet no LoRA applied — fine-tuning a 3B model without PEFT is VRAM-heavy
- ⚠️ 4-GPU memory assumption

---

### `src/model/gnn.py`
- **Role**: GNN implementations used as graph encoders
- `GCN(in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1)` — stacked `GCNConv` + `BatchNorm1d` + ReLU + dropout
- `GAT(in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4)` — stacked custom `GATConv` + `BatchNorm1d` + ReLU + dropout
- `load_gnn_model` dict: `{'gcn': GCN, 'gat': GAT}`
- Both return `(node_embeddings, edge_attr)`
- Imports: `torch_geometric.nn.GCNConv`, custom `GATConv`

---

### `src/model/gnn_layer/gat_layer.py` — `GATConv`
- **Role**: Custom GAT layer (copy of PyG's `GATConv`) with edge attribute support and modified `add_self_loops=False` default
- `forward(x, edge_index, edge_attr, size, return_attention_weights)` — full GAT forward with optional edge features
- `edge_update(alpha_j, alpha_i, edge_attr, ...)` — computes and normalizes attention weights
- `message(x_j, alpha)` — weighted message passing
- Imports: `torch_geometric` (MessagePassing, Linear, SparseTensor utils)
- Stores edge_attr in `self.e` (mutable instance state during forward) — not thread-safe

---

### `src/model/gnn_layer/gated_gcn_layer.py` — `GatedGCNLayer`
- **Role**: Gated GCN layer with edge features and residual connections (from arXiv:1711.07553)
- `forward(x, e, edge_index)` — node + edge feature update via sigmoid gating
- `message / aggregate / update` — custom gated aggregation
- Imports: `torch_geometric.nn`, `torch_scatter.scatter`
- **Never imported or used anywhere in the main pipeline** — dead code

---

### `src/model/llama/model.py`
- **Role**: From-scratch LLaMA-1 Transformer with adapter support (Meta's original code)
- `ModelArgs` — dataclass with model hyperparams including `adapter_len`, `adapter_layer`
- `RMSNorm` — root mean square layer norm
- `Attention` — multi-head attention with RoPE; adapter prefix appended to keys/values in last `adapter_layer` layers
- `FeedForward` — SwiGLU FFN
- `TransformerBlock` — attention + FFN + adapter
- `Transformer` — full model with `forward(tokens)` and `forward_only(tokens, start_pos)` for KV-cache decoding
- Imports: `torch`, standard Python
- Only used by `LlamaAdapter`; the HuggingFace models in `GraphLLM`/`LLM`/`T5` do not use this

---

### `src/model/llama/generation.py` — `LLaMA`
- **Role**: Wrapper class holding `Transformer` + `Tokenizer`; `generate()` method is **entirely commented out**
- Unused in practice — generation logic is inlined in `LlamaAdapter.inference()`

---

### `src/model/llama/tokenizer.py` — `Tokenizer`
- **Role**: SentencePiece tokenizer wrapper for original LLaMA-1
- `encode(s, bos, eos)` / `decode(t)` — tokenize/detokenize strings
- Used only by `LlamaAdapter`

---

### `src/dataset/__init__.py`
- **Role**: Dataset registry
- `load_dataset` dict: `{'cora': CoraDataset, 'citeseer': CiteseerDataset, 'pubmed': PubmedDataset, 'arxiv': ArxivDataset, 'products': ProductsDataset}`

---

### `src/dataset/cora.py` — `CoraDataset`
- **Role**: PyTorch Dataset for Cora (2,708 nodes, 7 classes)
- `__getitem__(index)` — returns `{id, label, desc (title+abstract), question}`
- `get_idx_split()` — loads train/val/test indices from txt files
- Features: 1433-dim bag-of-words; prompt: 7-class subcategory question
- Hardcoded paths relative to CWD — must be run from project root

---

### `src/dataset/citeseer.py` — `CiteseerDataset`
- **Role**: PyTorch Dataset for Citeseer (3,327 nodes, 6 classes)
- Same structure as `CoraDataset`; `desc` uses only abstract (no title)
- Features: 3703-dim; prompt: 6-category classification question

---

### `src/dataset/pubmed.py` — `PubmedDataset`
- **Role**: PyTorch Dataset for PubMed (19,717 nodes, 3 classes)
- `desc` = title + abstract; diabetes-type classification prompt
- ⚠️ `torch.load` without `weights_only=False` — will warn on PyTorch ≥ 2.0 and error on future versions

---

### `src/dataset/arxiv.py` — `ArxivDataset`
- **Role**: PyTorch Dataset for ogbn-arxiv (169,343 nodes, 40 classes)
- `__getitem__` also returns `full_label` (human-readable category name) in addition to `label` (cs.XX)
- ⚠️ `torch.load` without `weights_only=False`

---

### `src/dataset/products.py` — `ProductsDataset`
- **Role**: PyTorch Dataset for ogbn-products (~316K nodes, 47 classes)
- `__init__` converts `adj_t` (SparseTensor) to dense then extracts `edge_index` — extremely memory-intensive for 316K nodes (~100B entries)
- `graph_type = 'Product co-purchasing network'` (not a TAG)
- `__getitem__` uses `index` directly (not `text['node_id']`) as the node id — inconsistent with other datasets
- ⚠️ `adj_t.to_dense()` on a 316K×316K sparse matrix will OOM on most machines

---

### `src/dataset/preprocess/cora.py`
- **Role**: One-time preprocessing script for Cora
- `parse_cora()` — reads `.content` and `.cites` files, builds X/Y/edges
- `get_raw_text_cora()` — reads title/abstract from McCallum extraction files
- `preprocess()` — saves `data.pt` + `text.csv` + train/val/test split

---

### `src/dataset/preprocess/citeseer.py`
- **Role**: One-time preprocessing for Citeseer; reads `citeseer_texts.txt`
- Maps short class codes (`'AI'`) to full names (`'Artificial Intelligence'`)
- ⚠️ Uses `pd.merge(how='outer')` for text alignment — nodes with missing text get `'None'` string as abstract

---

### `src/dataset/preprocess/pubmed.py`
- **Role**: One-time preprocessing for PubMed-Diabetes; parses tab-separated node/edge files
- `parse_pubmed()` — builds 19717×500 feature matrix + adjacency
- `preprocess()` — saves graph + text CSV + split

---

### `src/dataset/preprocess/arxiv.py`
- **Role**: One-time preprocessing for ogbn-arxiv; uses OGB's official split (not random 60/20/20)
- Merges `nodeidx2paperid.csv.gz` with `titleabs.tsv` to get text

---

### `src/dataset/preprocess/products.py`
- **Role**: One-time preprocessing for ogbn-products
- ⚠️ `classes[24] = 'None'` (Python `None`, not the string `'NaN'`) — but `products.py` dataset class calls `fillna('NaN')` and `evaluate.py` checks for `'NaN'` in the class list; the preprocess script would write `'None'` to the CSV while the eval expects `'NaN'`
- ⚠️ Saves CSV with column `'nid'` but the dataset loader reads column `'node_id'` — will crash at load time

---

### `src/dataset/preprocess/generate_split.py`
- **Role**: Generates 60/20/20 train/val/test splits and saves to txt files
- `generate_split(num_nodes, path)` — sklearn `train_test_split`, fixed seed 42

---

### `src/utils/collate.py`
- **Role**: Custom DataLoader collate function — samples k-hop subgraphs on the fly
- `batch_subgraph(edge_index, node_ids, num_nodes, num_hops=3, fans_out=(50,50,50))` — random neighborhood sampling per node, re-labels and batches subgraphs
- `TAGCollator.__call__(original_batch)` — extracts features, labels, edge_index for batch of subgraphs; returns `{x, y, edge_index, mapping, batch, id, label, desc, question}`
- `collate_funcs` dict: all 5 datasets map to `TAGCollator`
- ⚠️ `fans_out` is hard-coded to `(50, 50, 50)` and not configurable; node sampling is random (no seed), so batches are non-deterministic even with fixed seed

---

### `src/utils/evaluate.py`
- **Role**: Post-processing and accuracy calculation for each dataset
- `get_accuracy_cora(eval_output, path)` — regex match on generated text vs class list; saves JSONL to path
- `get_accuracy_citeseer(eval_output, path)` — substring match (`label in pred`)
- `get_accuracy_pubmed(eval_output, path)` — substring match
- `get_accuracy_arxiv(eval_output, path)` — regex `cs\.[a-z]{2}` on first match
- `get_accuracy_products(eval_output, path)` — regex match against full class list
- `eval_funcs` dict maps dataset names to functions
- ⚠️ Substring/first-match heuristics are fragile — LLM can prefix correct answer with text that matches a wrong class; no normalization/whitespace stripping for Cora/Citeseer/PubMed
- ⚠️ `get_accuracy_pubmed` uses `label in pred` — partial matches cause false positives (e.g. "Type 1" in "Type 1 or Type 2 diabetes")

---

### `src/utils/ckpt.py`
- **Role**: Checkpoint save/load utilities
- `print_trainable_params(model)` — counts trainable params (standalone version, also duplicated inside each model class)
- `_save_checkpoint(model, optimizer, epoch, args, is_best)` — saves only trainable params (prunes frozen weights), adds optimizer state and config
- `_reload_best_model(model, args)` — loads `checkpoint_best.pth`; uses `strict=False`
- `_reload_model(model, checkpoint_path)` — same but arbitrary path
- ⚠️ `torch.load(..., weights_only=False)` explicitly disables security; safe here but noteworthy

---

### `src/utils/lr_schedule.py`
- **Role**: Cosine LR schedule with linear warmup
- `adjust_learning_rate(param_group, epoch, args)` — warmup for `args.warmup_epochs`, then half-cycle cosine decay to `args.min_lr`
- Contains a commented-out alternative version that iterates all param groups (the active version updates a single group)

---

### `src/utils/misc.py`
- **Role**: Distributed training utilities (copied from MAE/DeiT/BEiT)
- `SmoothedValue` — windowed/global metric tracking with deque
- `MetricLogger` — dict of `SmoothedValue` with DDP sync and `log_every` generator
- `NativeScalerWithGradNormCount` — AMP `GradScaler` wrapper with optional gradient clipping
- `init_distributed_mode(args)` — handles OMPI, env-var, and SLURM rank detection
- `all_reduce_mean(x)`, `save_model(...)`, `load_model(...)` — DDP helpers
- Copyright: Meta Platforms

---

### `src/utils/seed.py`
- **Role**: Global seed setter
- `seed_everything(seed)` — sets Python, NumPy, PyTorch, CUDA seeds; sets `cudnn.deterministic=True, benchmark=True`
- ⚠️ `benchmark=True` + `deterministic=True` is contradictory — benchmark mode selects fastest non-deterministic kernels; `deterministic=True` should suppress that but behavior depends on PyTorch version

---

### `brazil/generate_brazil_descriptions.py`
- **Role**: Standalone tool — generates TANS-style textual descriptions for Brazil Airport graph nodes using OpenAI API
- `compute_topology(pyg_root)` — loads PyG `Airports` dataset, computes 5 NetworkX centrality metrics per node
- `run_step1(...)` — compute or load topology; saves to `.npy`
- `build_prompt(node_idx, topo)` — builds few-shot prompt with hardcoded node-0 example
- `query_llm(client, model_name, user_prompt, max_retries)` — OpenAI chat completion with exponential backoff
- `run_step2(topo, client, model_name, output_csv, num_nodes)` — queries LLM for all nodes, writes CSV
- Imports: `networkx`, `openai`, `torch_geometric.datasets.Airports`, `numpy`
- ⚠️ **OPENAI_API_KEY is stored in `.env` as plaintext and `.env` is NOT in `.gitignore`** (see below)

---

## Architecture & Data Flow

### Main pipeline (GraphLLM, `train.py`)

```
Raw dataset (e.g. Cora)
    └─ preprocess/ scripts
           ├─ graph: data.pt (PyG Data: x, edge_index, y)
           └─ text.csv (node_id, label, title, abstract)

Dataset class (__getitem__)
    └─ {id, label, desc, question}   ← one item per node

TAGCollator (collate_fn)
    └─ batch_subgraph(edge_index, node_ids, num_hops=3, fans_out=50)
           ├─ x = graph.x[subset]         ← subgraph node features
           ├─ edge_index_sub              ← remapped subgraph edges
           └─ mapping                     ← index of target node in subgraph

GraphLLM.forward(batch)
    ├─ GNN (GAT, 4 layers, 1024-dim)
    │      x, edge_index → node embeddings [N_sub, 1024]
    │      n_embeds[mapping] → target node embedding [B, 1024]
    ├─ Projector MLP: [B, 1024] → [B, 2048] → Sigmoid → [B, 4096]
    │      graph_embeds: [B, 4096]  (1 token per node)
    ├─ Tokenizer (Vicuna-7B):
    │      desc[i][:512 tokens] + question[i] + label[i] + EOS
    │      → token IDs → word_embedding → [seq_len, 4096]
    └─ LLM input: [BOS][graph_token][desc_tokens][question_tokens][label_tokens]
           → causal LM → cross-entropy loss on label tokens only

GraphLLM.inference(batch)
    └─ same assembly minus label tokens → model.generate() → decode → class string
```

### Model variants

| Model | Graph token | LLM backbone | Trainable |
|---|---|---|---|
| `GraphLLM` (frozen) | GNN+projector | Vicuna-7B frozen | GNN + projector (~8M) |
| `GraphLLM` (LoRA) | GNN+projector | Vicuna-7B + LoRA(r=8) | GNN + projector + LoRA |
| `LLM` | None | Vicuna-7B | LoRA only (or frozen) |
| `PromptTuningLLM` | Learned soft tokens | Vicuna-7B frozen | `prompt` embedding |
| `T5` | None | FLAN-T5-XL | All params (no PEFT) |
| `LlamaAdapter` | None | LLaMA-1 7B | adapter/gate layers |
| `GCN`/`GAT` | — | None | All GNN params (train_gnn.py) |

---

## Key Observations & Warnings

### 🔴 Critical

1. **Exposed API key**: `.env` contains a live OpenAI API key (`sk-proj-...`) and `.gitignore` does **not** exclude `.env` — the key is committed to the repository and visible to anyone with access
2. **`ProductsDataset.__init__` OOM**: `adj_t.to_dense()` on a 316K×316K sparse matrix allocates ~100B float32 entries — will crash on any normal machine
3. **`inference.py` loads no checkpoint**: There is no `_reload_best_model` call — model weights are random at test time unless a checkpoint is explicitly provided (no `--resume` handling)

### 🟠 Bugs

4. **`train_gnn.py:83` wrong call signature**: `adjust_learning_rate(param_group, args.lr, epoch_frac, args)` passes 4 args; function takes 3 (`param_group, epoch, args`) — `args.lr` is used as epoch, real epoch value as `args`; LR schedule is broken
5. **`best_epoch` used before assignment** in `train.py:107` and `train_gnn.py:121` — `NameError` on first epoch if the best-model condition is never hit
6. **`products` preprocess/eval class mismatch**: `preprocess/products.py` sets `classes[24] = 'None'` (Python `None`), but `evaluate.py` lists `'NaN'` as class 24's name; the stored CSV label will not match
7. **`products` preprocess CSV column name mismatch**: saves with column `'nid'` but `ProductsDataset` reads `'node_id'` — crash at load
8. **`PromptTuningLLM.encode_graphs`** references `self.graph_encoder` and `self.graph` — neither is ever assigned; will raise `AttributeError` if called
9. **`LlamaAdapter.forward` attention mask commented out** — model attends over pad tokens during training
10. **`LlamaAdapter` ignore_index = 0** — token 0 is typically BOS/pad, not a true ignore marker; loss is computed differently than intended vs `-100`

### 🟡 Design / Quality

11. **`train_ddp.py` uses `sampler_val` for test loader** — test and val loaders share the same distributed sampler; test batches may be distributed incorrectly
12. **Projector uses `Sigmoid`** between two linear layers — kills gradients for large inputs; ReLU/GELU would be standard for a cross-modal alignment projector
13. **`GatedGCNLayer` is dead code** — implemented and imported into the directory but never referenced in any model
14. **`llama/generation.py` generate() method is entirely commented out** — `LLaMA` class is unused
15. **`--llm_frozen` is a string flag** compared with `== 'True'` — passing `--llm_frozen 1` or `--llm_frozen true` silently keeps the model frozen
16. **`seed_everything` sets `benchmark=True` and `deterministic=True` simultaneously** — contradictory; benchmark mode picks fastest non-deterministic kernels
17. **`fans_out` hard-coded to `(50,50,50)`** in `TAGCollator` — not configurable; subgraph size directly affects GPU memory and cannot be tuned from CLI
18. **`LLM`/`PromptTuningLLM`/`T5` accept `graph`/`graph_type`/`prompt` but ignore them** — misleading signatures; should be `**kwargs` or removed
19. **`print_trainable_params` duplicated** in `ckpt.py` and in every model class — should be a shared utility
20. **Evaluation metrics are text-matching heuristics** — LLM output parsing is fragile (first regex match, substring containment); any preamble before the class name can cause false negatives or false positives
21. **All LLM model files hard-code multi-GPU memory dicts** (`{0:'20GiB', ...}`) — should use `device_map='auto'` with a configurable `max_memory` arg
