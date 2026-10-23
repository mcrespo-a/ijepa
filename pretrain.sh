#!/bin/bash
#SBATCH --job-name=ijepa_vitsmall
#SBATCH --partition=dynamic-64cores-256g-1gpu-A100-40g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/project/home/mcrespoa/ijepa-freq/logs/vitsmall/%x_%j.out
#SBATCH --error=/project/home/mcrespoa/ijepa-freq/logs/vitsmall/%x_%j.err

set -euo pipefail

IJEPA_ENV="${IJEPA_ENV:-/project/home/mcrespoa/envs/vechoj}"
# IJEPA_ENV="${IJEPA_ENV:-/project/home/mcrespoa/envs/vechoj-pascal}"
source "${IJEPA_ENV}/bin/activate"

# -- Paths
PROJECT_DIR="/project/home/mcrespoa/ijepa-freq"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/radimagenet/pretrain.yaml}"
ENTRY="main.py"

cd "$PROJECT_DIR"
mkdir -p "${PROJECT_DIR}/logs/vitsmall"  # the #SBATCH --output dir must also exist BEFORE sbatch

# -- Threads: keep BLAS/OMP from oversubscribing against dataloader workers
export OMP_NUM_THREADS=1

# -- SLURM 22.05+ no longer propagates --cpus-per-task to srun; set it explicitly
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

# -- Per-job rendezvous port so concurrent jobs on a shared node do not collide
PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))

# -- wandb: offline by default (config wandb.mode), written under logging.folder;
#    sync afterwards with `wandb sync <logging.folder>/wandb/offline-run-*`

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_NTASKS=${SLURM_NTASKS}  PORT=${PORT}"
echo "CONFIG_PATH=${CONFIG_PATH}"
nvidia-smi

# main.py spawns one process per --devices entry (single GPU here) and sets up
# torch.distributed itself, so srun launches a single task.
srun --kill-on-bad-exit=1 \
    python -u "${ENTRY}" \
        --fname "${CONFIG_PATH}" \
        --devices cuda:0 \
        --port "${PORT}"

echo "Job finished at $(date)"
