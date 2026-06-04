#!/bin/bash
# ============================================================
# Airport cross-dataset experiments
#   Set A: train Europe  → test Brazil  (seeds 0, 1, 2)
#   Set B: train Brazil  → test Europe  (seeds 0, 1, 2)
# ============================================================

set -e  # stop on first error

# Set to "disabled" to skip wandb logging, "online" to log to wandb cloud
export WANDB_MODE=online

SEEDS=(0 1 2)
MODEL=graph_llm
LLM=7b
GNN=gat
LOG_DIR=logs/airport
mkdir -p "$LOG_DIR"

# ── Prerequisites check ──────────────────────────────────────
echo "Checking preprocessed datasets..."
for name in europe brazil; do
    pt="dataset/tape_${name}/processed/data.pt"
    if [ ! -f "$pt" ]; then
        echo "  ERROR: $pt not found."
        echo "  Run first:"
        echo "    python generate_descriptions.py --dataset $name"
        echo "    python -m src.dataset.preprocess.airports --dataset $name"
        exit 1
    fi
done
echo "  OK — both datasets ready."
echo ""

# ── Helper function ──────────────────────────────────────────
run_experiment() {
    local train_ds=$1
    local test_ds=$2
    local seed=$3
    local tag="${train_ds}→${test_ds}_seed${seed}"
    local log="${LOG_DIR}/${tag}.log"

    echo "────────────────────────────────────────────────────────"
    echo "  Run : $tag"
    echo "  Log : $log"
    echo "────────────────────────────────────────────────────────"

    python train.py \
        --dataset      "$train_ds" \
        --test_dataset "$test_ds"  \
        --model_name   "$MODEL"    \
        --llm_model_name "$LLM"    \
        --gnn_model_name "$GNN"    \
        --seed         "$seed"     \
        2>&1 | tee "$log"

    echo ""
    echo "  Done: $tag"
    echo ""
}

# ── Set A: Europe → Brazil ───────────────────────────────────
echo "========================================================"
echo "  SET A — Train: Europe   Test: Brazil"
echo "========================================================"
for seed in "${SEEDS[@]}"; do
    run_experiment europe brazil "$seed"
done

# ── Set B: Brazil → Europe ───────────────────────────────────
echo "========================================================"
echo "  SET B — Train: Brazil   Test: Europe"
echo "========================================================"
for seed in "${SEEDS[@]}"; do
    run_experiment brazil europe "$seed"
done

# ── Summary ──────────────────────────────────────────────────
echo "========================================================"
echo "All runs complete. Results summary:"
echo "========================================================"
for f in "$LOG_DIR"/*.log; do
    tag=$(basename "$f" .log)
    acc=$(grep "Test Acc" "$f" | tail -1 | grep -oP '[0-9]+\.[0-9]+' | tail -1)
    printf "  %-35s  Test Acc: %s\n" "$tag" "${acc:-N/A}"
done
