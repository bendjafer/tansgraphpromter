#!/bin/bash
# ============================================================
# Airport experiments — TANS paper setup (Table 13)
#
# Data split  : source = 100% train
#               target = 20% val  (model selection by val acc)
#                        80% test (reported accuracy)
#
# Backbone    : GCN only
#
# Hyperparams (per direction, from TANS Table 13):
#   Brazil → Europe : hidden=16, layers=3, normalize=none,      lr=5e-2, wd=1e-4, drop=0.0
#   Europe → Brazil : hidden=32, layers=2, normalize=batchnorm, lr=5e-3, wd=0,    drop=0.8
#
# Experiments : TPF  (5 topo features, standardised)
#               MiniLM (384-dim embeddings)
#
# Seeds       : 0 1 2  — mean ± std reported at end
# ============================================================

set -e

# Load WANDB_API_KEY from .env if present
_WANDB_KEY=$(python -c "
import re, pathlib
try:
    text = pathlib.Path('.env').read_text()
    m = re.search(r'WANDB_API_KEY\s*=\s*[\"\']*([^\"\'\\n]+)[\"\']*', text)
    v = m.group(1).strip() if m else ''
    print(v)
except:
    print('')
" 2>/dev/null)

if [ -n "$_WANDB_KEY" ]; then
    export WANDB_API_KEY="$_WANDB_KEY"
    export WANDB_MODE=online
    echo "wandb: online (key loaded from .env)"
else
    export WANDB_MODE=disabled
    echo "wandb: disabled (no WANDB_API_KEY in .env)"
fi

SEEDS=(0 1 2)
MODEL=graph_llm
LLM=7b
GNN=gcn
OUTPUT_DIR=output_tans
LOG_DIR=logs/airport_tans
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ── Prerequisites ────────────────────────────────────────────
echo "Checking preprocessed datasets..."
for name in europe brazil europe_minilm brazil_minilm; do
    pt="dataset/tape_${name}/processed/data.pt"
    if [ ! -f "$pt" ]; then
        echo "  ERROR: $pt not found. Run preprocessing first."
        exit 1
    fi
done
echo "  OK — all datasets ready."
echo ""

# ── Helper ───────────────────────────────────────────────────
run_experiment() {
    local train_ds="$1"
    local test_ds="$2"
    local seed="$3"
    local hidden="$4"
    local layers="$5"
    local normalize="$6"
    local lr="$7"
    local wd="$8"
    local drop="$9"
    local norm_feat="${10}"
    local tag="${train_ds}→${test_ds}_seed${seed}"
    local log="${LOG_DIR}/${tag}.log"
    local norm_flag="--gnn_normalize $normalize"

    # Build normalize_features flag
    local feat_flag=""
    if [ "$norm_feat" = "--normalize_features" ]; then
        feat_flag="--normalize_features"
    fi

    echo "────────────────────────────────────────────────────────"
    echo "  Run : $tag"
    echo "  GCN : hidden=${hidden} layers=${layers} norm=${normalize} drop=${drop}"
    echo "  Opt : lr=${lr} wd=${wd}"
    echo "  Log : $log"
    echo "────────────────────────────────────────────────────────"

    python train.py \
        --dataset           "$train_ds"   \
        --test_dataset      "$test_ds"    \
        --model_name        "$MODEL"      \
        --llm_model_name    "$LLM"        \
        --gnn_model_name    "$GNN"        \
        --gnn_hidden_dim    "$hidden"     \
        --gnn_out_dim       "$hidden"     \
        --gnn_num_layers    "$layers"     \
        --gnn_dropout       "$drop"       \
        --lr                "$lr"         \
        --wd                "$wd"         \
        --target_val_ratio  0.2           \
        --output_dir        "$OUTPUT_DIR" \
        --seed              "$seed"       \
        $norm_flag                        \
        $feat_flag                        \
        2>&1 | tee "$log"

    echo ""
    echo "  Done: $tag"
    echo ""
}

# ────────────────────────────────────────────────────────────
# TPF experiments  (--normalize_features)
# ────────────────────────────────────────────────────────────
echo "========================================================"
echo "  TPF — Train: brazil   Test: europe"
echo "  hidden=16, layers=3, normalize=none, lr=5e-2, wd=1e-4, drop=0"
echo "========================================================"
for seed in "${SEEDS[@]}"; do
    run_experiment brazil europe "$seed" 16 3 none 5e-2 1e-4 0.0 "--normalize_features"
done

echo "========================================================"
echo "  TPF — Train: europe   Test: brazil"
echo "  hidden=32, layers=2, normalize=batchnorm, lr=5e-3, wd=0, drop=0.8"
echo "========================================================"
for seed in "${SEEDS[@]}"; do
    run_experiment europe brazil "$seed" 32 2 batchnorm 5e-3 0 0.8 "--normalize_features"
done

# ────────────────────────────────────────────────────────────
# MiniLM experiments  (no feature normalization)
# ────────────────────────────────────────────────────────────
echo "========================================================"
echo "  MiniLM — Train: brazil_minilm   Test: europe_minilm"
echo "  hidden=16, layers=3, normalize=none, lr=5e-2, wd=1e-4, drop=0"
echo "========================================================"
for seed in "${SEEDS[@]}"; do
    run_experiment brazil_minilm europe_minilm "$seed" 16 3 none 5e-2 1e-4 0.0 ""
done

echo "========================================================"
echo "  MiniLM — Train: europe_minilm   Test: brazil_minilm"
echo "  hidden=32, layers=2, normalize=batchnorm, lr=5e-3, wd=0, drop=0.8"
echo "========================================================"
for seed in "${SEEDS[@]}"; do
    run_experiment europe_minilm brazil_minilm "$seed" 32 2 batchnorm 5e-3 0 0.8 ""
done

# ────────────────────────────────────────────────────────────
# Summary — mean ± std per direction × feature set
# ────────────────────────────────────────────────────────────
echo "========================================================"
echo "All TANS runs complete."
echo "========================================================"

python - <<'PYEOF'
import os, re, math, glob

log_dir = "logs/airport_tans"

groups = {}
for f in sorted(glob.glob(f"{log_dir}/*.log")):
    tag = os.path.basename(f).replace(".log", "")
    group = re.sub(r"_seed\d+$", "", tag)
    content = open(f).read()
    matches = re.findall(r"Test Acc[^:]*:\s*([0-9]+\.[0-9]+)", content)
    if matches:
        groups.setdefault(group, []).append(float(matches[-1]))

print(f"\n{'Group':<50}  Seeds  Mean    Std")
print("-" * 75)
for group, accs in sorted(groups.items()):
    mean = sum(accs) / len(accs)
    std  = math.sqrt(sum((a - mean)**2 for a in accs) / len(accs))
    seeds_str = str(len(accs))
    print(f"  {group:<48}  {seeds_str}      {mean:.4f}  ±{std:.4f}")
PYEOF