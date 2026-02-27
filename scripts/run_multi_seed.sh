#!/bin/bash
# Run multi-seed training on available GPUs

SEEDS=(42 43 44 45 46)
SPLITS=(target_unseen e3_unseen random)
GPUS=(6 7 8 9)

# Function to run training
run_training() {
    seed=$1
    split=$2
    gpu=$3

    echo "[$(date)] Starting seed=$seed, split=$split on GPU $gpu"

    CUDA_VISIBLE_DEVICES=$gpu python scripts/train.py \
        --phase 2 \
        --splits $split \
        --finetune-epochs 20 \
        --class-weights \
        --seed $seed \
        > /tmp/train_seed${seed}_${split}.log 2>&1

    # Copy checkpoint with seed suffix
    if [ -f "checkpoints/phase2_${split}/best_model.pt" ]; then
        cp "checkpoints/phase2_${split}/best_model.pt" "checkpoints/phase2_${split}/best_model_seed${seed}.pt"
    fi

    echo "[$(date)] Completed seed=$seed, split=$split"
}

# Run in batches to use all GPUs
job_idx=0
for seed in "${SEEDS[@]}"; do
    for split in "${SPLITS[@]}"; do
        gpu_idx=$((job_idx % ${#GPUS[@]}))
        gpu=${GPUS[$gpu_idx]}

        run_training $seed $split $gpu &

        job_idx=$((job_idx + 1))

        # Every 4 jobs, wait for completion
        if [ $((job_idx % 4)) -eq 0 ]; then
            wait
            echo "[$(date)] Batch completed, continuing..."
        fi
    done
done

wait
echo "[$(date)] All training complete!"
