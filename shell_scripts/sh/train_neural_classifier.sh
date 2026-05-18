#!/bin/bash

export CUDA_MATMUL_TF32='high'
export LOG_LEVEL='INFO'
export WARNING_LOG_FILE='train.err'

# Default number of runs
runs=10
exp_name="${EXP_NAME:-finetune}"
args=()

# Handle Ctrl+C
trap "echo 'Interrupted! Stopping all processes...'; exit 1" SIGINT

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --runs) runs="$2"; shift 2 ;;
        --exp_name|--exp-name) exp_name="$2"; shift 2 ;;
        --exp_name=*) exp_name="${1#*=}"; shift ;;
        --exp-name=*) exp_name="${1#*=}"; shift ;;
        *) args+=("$1"); shift ;;
    esac
done

# Loop over the number of runs
for ((i=0; i<runs; i++)); do
    wandb_run_name="${exp_name}_seed_${i}"
    echo "Running finetuning seed $i as $wandb_run_name"
    python ./asymdsd/run/classification_cli.py fit \
    --config configs/classification/classification.yaml \
    "${args[@]}" \
    --seed_everything "$i" \
    --trainer.logger.init_args.name "$wandb_run_name"
done
