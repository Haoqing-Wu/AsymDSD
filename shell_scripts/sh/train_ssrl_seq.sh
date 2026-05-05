#!/bin/bash

export CUDA_MATMUL_TF32='high'
export LOG_LEVEL='INFO'
export WARNING_LOG_FILE='train.err'

python ./asymdsd/run/ssrl_seq_cli.py fit \
    --config configs/ssrl/ssrl_seq.yaml \
    --data.init_args.multi_crop_config=null \
    "$@"
