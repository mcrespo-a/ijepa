#!/bin/bash
#SBATCH --job-name=ijepa_vitsmall_mn
#SBATCH --partition=dynamic-64cores-256g-1gpu-A100-40g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # = GPUs per node: one training process per GPU
#SBATCH --cpus-per-task=8            # per GPU; keep data.num_workers <= this
##SBATCH --gres=gpu:1                # uncomment (= ntasks-per-node) if the partition does not allocate GPUs implicitly
#SBATCH --output=/project/home/mcrespoa/ijepa/logs/vitsmall/%x_%j.out
#SBATCH --error=/project/home/mcrespoa/ijepa/logs/vitsmall/%x_%j.err

# Multi-node (and/or multi-GPU-per-node) I-JEPA pretraining.
#
#   sbatch pretrain_multinode.sh
#   sbatch --nodes=4 pretrain_multinode.sh                        # 4 nodes x 1 GPU
#   sbatch --nodes=2 --ntasks-per-node=4 --gres=gpu:4 \
#          --partition=<4-gpu partition> pretrain_multinode.sh    # 2 nodes x 4 GPUs
#
# srun starts one task per GPU on every node; each task runs main_multinode.py,
# which reads its rank from SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID and
# rendezvouses at MASTER_ADDR:MASTER_PORT (first node of the allocation).
#
# data.batch_size is PER GPU: the global batch is batch_size * nodes * ntasks-per-node.
# The lr in the config is not rescaled automatically; adjust it if you change the
# global batch size w.r.t. the single-GPU run.

set -euo pipefail

# -- Environment (see pretrain.sh)
IJEPA_ENV="${IJEPA_ENV:-/project/home/mcrespoa/envs/vechoj}"
source "${IJEPA_ENV}/bin/activate"

# -- Paths
PROJECT_DIR="/project/home/mcrespoa/ijepa"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/radimagenet/pretrain.yaml}"
ENTRY="main_multinode.py"

cd "$PROJECT_DIR"
mkdir -p "${PROJECT_DIR}/logs/vitsmall"  # the #SBATCH --output dir must also exist BEFORE sbatch

# -- Threads: keep BLAS/OMP from oversubscribing against dataloader workers
export OMP_NUM_THREADS=1

# -- SLURM 22.05+ no longer propagates --cpus-per-task to srun; set it explicitly
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

# -- Rendezvous: first node of the allocation, per-job port so concurrent jobs
#    on shared nodes do not collide. Exported, so every srun task inherits them.
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))

# -- NCCL: surface init/transport errors in the .err log. If the nodes have
#    several network interfaces and NCCL picks the wrong one (hangs at init),
#    pin it, e.g. export NCCL_SOCKET_IFNAME=eth0 (check `ip addr` on a node).
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# -- wandb: offline by default (config wandb.mode), only rank 0 logs;
#    sync afterwards with `wandb sync <logging.folder>/wandb/offline-run-*`

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  NODES=${SLURM_JOB_NUM_NODES} (${SLURM_JOB_NODELIST})"
echo "SLURM_NTASKS=${SLURM_NTASKS}  NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
echo "CONFIG_PATH=${CONFIG_PATH}"

# -- GPUs on every node (one line per node)
srun --ntasks-per-node=1 --ntasks="${SLURM_JOB_NUM_NODES}" \
    bash -c 'echo "$(hostname): $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | paste -sd ";" -)"'

srun --kill-on-bad-exit=1 \
    python -u "${ENTRY}" \
        --fname "${CONFIG_PATH}" \
        --port "${MASTER_PORT}"

echo "Job finished at $(date)"
