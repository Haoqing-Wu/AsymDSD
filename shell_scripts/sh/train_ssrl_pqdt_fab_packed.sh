#!/bin/bash

export CUDA_MATMUL_TF32='high'
export LOG_LEVEL='INFO'
export WARNING_LOG_FILE='train.err'

python ./asymdsd/run/ssrl_pqdt_fab_packed_cli.py fit \
    --config configs/ssrl/ssrl_pqdt_fab_packed.yaml \
    "$@"
