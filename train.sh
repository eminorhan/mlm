#!/bin/bash

#SBATCH --account=stf218
#SBATCH --nodes=64
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=train_mlm
#SBATCH --output=train_mlm_%A_%a.out
#SBATCH --array=0
#SBATCH --qos=debug

# set proxy server to enable communication with outside
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

# set misc env vars
export LOGLEVEL=INFO
export LD_LIBRARY_PATH=/lustre/orion/stf218/scratch/emin/aws-ofi-rccl/lib:$LD_LIBRARY_PATH  # enable aws-ofi-rccl
export NCCL_NET_GDR_LEVEL=3   # can improve performance, but remove this setting if you encounter a hang/crash.
export NCCL_ALGO=TREE         # may see performance difference with either setting. (should not need to use this, but can try)
export NCCL_CROSS_NIC=1       # on large systems, this nccl setting has been found to improve performance
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_IB_TIMEOUT=31
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCHELASTIC_ENABLE_FILE_TIMER=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="/lustre/orion/stf218/scratch/emin/huggingface"
export HF_DATASETS_CACHE="/lustre/orion/stf218/scratch/emin/huggingface"
export HF_HUB_OFFLINE=1
export GPUS_PER_NODE=8

# set network
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=3442

# root model directory
MODEL_ROOT_DIR="/lustre/orion/stf218/scratch/emin/mlm/models"
SP="xlm-roberta-large"

export GPUS_PER_NODE=8

export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_NODEID \
    --mixed_precision no \
    --rdzv_backend c10d \
    --use_fsdp \
    --fsdp_auto_wrap_policy SIZE_BASED_WRAP \
    --fsdp_backward_prefetch BACKWARD_PRE \
    --fsdp_min_num_params 2000 \
    --fsdp_sharding_strategy 1 \
    --fsdp_state_dict_type FULL_STATE_DICT \
    "
export SCRIPT="/lustre/orion/stf218/scratch/emin/mlm/train.py"
export SCRIPT_ARGS=" \
    --config_name "answerdotai/ModernBERT-large" \
    --dataset_name "allenai/c4" \
    --dataset_config_name "realnewslike" \
    --max_seq_length 8192 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 0.0003 \
    --output_dir "${MODEL_ROOT_DIR}/${SP}" \
    --num_train_epochs 20 \
    --checkpointing_steps 100 \
    "
# this step is necessary because accelerate launch does not seem to handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
srun $CMD

echo "Done"