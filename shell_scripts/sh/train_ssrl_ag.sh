#!/bin/bash

export CUDA_MATMUL_TF32='high'
export LOG_LEVEL='INFO'
export WARNING_LOG_FILE='train.err'

python ./asymdsd/run/ssrl_ag_cli.py fit \
    --config configs/ssrl/ssrl.yaml \
    --model configs/ssrl/variants/model/ssrl_model_ag.yaml \
    "$@"
