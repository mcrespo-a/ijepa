# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Multi-node / multi-GPU entry point: the launcher (srun, see
# pretrain_multinode.sh, or torchrun) starts ONE process per GPU and this
# script joins the process group from the launcher's environment, whereas
# main.py spawns the per-GPU processes itself on a single node.
#

import argparse
import logging
import os
import pprint
import sys


from src.utils.distributed import init_distributed, get_local_rank
from src.config import load_config
from src.train import main as app_main

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--port', type=int, default=40112,
    help='rendezvous port, used only if MASTER_PORT is not exported')


def main():
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger()

    world_size, rank = init_distributed(port=args.port)
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    logger.info(f'Running... (rank: {rank}/{world_size}, local rank: {get_local_rank()}, '
                f'master: {os.environ.get("MASTER_ADDR")}:{os.environ.get("MASTER_PORT")})')
    logger.info(f'called-params {args.fname}')

    # -- load script params
    params = load_config(args.fname)
    if rank == 0:
        logger.info('loaded params...')
        pprint.PrettyPrinter(indent=4).pprint(params)

    app_main(args=params)


if __name__ == '__main__':
    main()
