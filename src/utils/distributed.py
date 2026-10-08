# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import subprocess

import torch
import torch.distributed as dist

from logging import getLogger

logger = getLogger()


def get_local_rank():
    """ Rank of this process on its node (torchrun or srun), 0 otherwise """
    for key in ('LOCAL_RANK', 'SLURM_LOCALID'):
        if key in os.environ:
            return int(os.environ[key])
    return 0


def _slurm_master_addr():
    """ First hostname of the job allocation (same on every node) """
    nodelist = os.environ.get('SLURM_JOB_NODELIST', os.environ.get('SLURM_NODELIST'))
    if nodelist:
        try:
            out = subprocess.check_output(
                ['scontrol', 'show', 'hostnames', nodelist], text=True)
            return out.split()[0]
        except Exception as e:
            logger.info(f'could not expand SLURM nodelist {nodelist}: {e}')
    return os.environ.get('HOSTNAME', 'localhost')


def init_distributed(port=40112, rank_and_world_size=(None, None)):
    """ Join (or create) the default nccl process group.

    rank_and_world_size given: processes spawned locally (main.py), rendezvous
    on localhost:port. Otherwise rank/world-size come from the launcher, one
    process per GPU: torchrun (RANK/WORLD_SIZE/MASTER_*) or srun (SLURM_PROCID/
    SLURM_NTASKS; MASTER_ADDR/MASTER_PORT from the env if exported, else the
    first node of the allocation and `port`).
    """

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size

    if (rank is None) or (world_size is None):
        if ('RANK' in os.environ) and ('WORLD_SIZE' in os.environ):
            world_size = int(os.environ['WORLD_SIZE'])
            rank = int(os.environ['RANK'])
        elif 'SLURM_PROCID' in os.environ:
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
        else:
            logger.info('SLURM vars not set (distributed training not available)')
            world_size, rank = 1, 0
            return world_size, rank
        os.environ.setdefault('MASTER_ADDR', _slurm_master_addr())
        os.environ.setdefault('MASTER_PORT', str(port))
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)

    try:
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank)
    except Exception as e:
        # -- a silent fallback would leave every rank training alone (and
        # -- all of them writing rank-0 checkpoints), so only allow it for 1 proc
        if world_size > 1:
            raise
        world_size, rank = 1, 0
        logger.info(f'distributed training not available {e}')

    return world_size, rank


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
