#!/bin/bash
# ============================================================
# Airport cross-dataset experiments — MiniLM features
#   x = MiniLM-L12-v2 embeddings of node descriptions [N, 384]
#   (compare directly against run_airport_experiments.sh which uses [N, 5] topo)
#
#   Set A: train europe_minilm → test brazil_minilm  (seeds 0, 1, 2)
#   Set B: train brazil_minilm → test europe_minilm  (seeds 0, 1, 2)
# ============================================================

set -e

export WANDB_MODE=disabled

SEEDS=(0 1 2)
MODEL=graph_llm
LLM=7b
GNN=gat
OUTPUT_DIR=output_minilm
LOG_DIR=logs/airport_minilmc
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ── Prerequisites check ──────────────────────────────────────
echo "Checking preprocessed MiniLM datasets..."
for name in europe_minilm brazil_minilm; do
    pt="dataset/tape_${name}/processed/data.pt"
    if [ ! -f "$pt" ]; then
        echo "  ERROR: $pt not found."
        echo "  Run first:"
        echo "    python -m src.dataset.preprocess.airports --dataset ${name%_minilm} --features minilm"
        exit 1
    fi
done
echo "  OK — both MiniLM datasets ready."
echo ""

# ── Helper ───────────────────────────────────────────────────
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
        --dataset        "$train_ds"  \
        --test_dataset   "$test_ds"   \
        --model_name     "$MODEL"     \
        --llm_model_name "$LLM"       \
        --gnn_model_name "$GNN"       \
        --output_dir     "$OUTPUT_DIR" \
        --seed           "$seed"      \
        2>&1 | tee "$log"

    echo ""
    echo "  Done: $tag"
    echo ""
}

# ── Set A: europe_minilm → brazil_minilm ─────────────────────
echo "========================================================"
echo "  SET A — Train: europe_minilm   Test: brazil_minilm"
echo "========================================================"
for seed in "${SEEDS[@]}"; do
    run_experiment europe_minilm brazil_minilm "$seed"
done

# ── Set B: brazil_minilm → europe_minilm ─────────────────────
echo "========================================================"
echo "  SET B — Train: brazil_minilm   Test: europe_minilm"
echo "========================================================"
for seed in "${SEEDS[@]}"; do
    run_experiment brazil_minilm europe_minilm "$seed"
done

# ── Summary ──────────────────────────────────────────────────
echo "========================================================"
echo "All MiniLM runs complete. Results summary:"
echo "========================================================"
for f in "$LOG_DIR"/*.log; do
    tag=$(basename "$f" .log)
    acc=$(grep "Test Acc" "$f" | tail -1 | grep -oP '[0-9]+\.[0-9]+' | tail -1)
    printf "  %-45s  Test Acc: %s\n" "$tag" "${acc:-N/A}"
done
