# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.masks.frequency import apply_freq_mask
from src.utils.distributed import (
    init_distributed,
    get_local_rank,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet1k
from src.datasets.radimgnet import make_radimgnet

from src.helper import (
    load_checkpoint,
    init_model,
    init_opt)
from src.transforms import make_transforms
from src.config import load_config

# --
log_timings = True
log_freq = 10
checkpoint_freq = 10
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

try:
    import wandb
except ImportError:
    wandb = None


def init_wandb(args, rank, folder):
    """ Start a single wandb run on rank 0. Never raises: returns True if live """
    cfg = args.get('wandb', None) or {}
    if rank != 0 or not cfg.get('enable', False):
        return False
    if wandb is None:
        logger.warning('[wandb] enabled in config but not installed, continuing without it')
        return False
    try:
        wandb.init(
            project=cfg.get('project', 'ijepa-pretrain'),
            entity=cfg.get('entity', None),
            group=cfg.get('group', None),
            name=cfg.get('run_name', None),
            id=cfg.get('run_id', None),
            config=args,
            resume='allow',
            mode=cfg.get('mode', 'offline'),
            dir=cfg.get('dir', None) or folder)
    except Exception as exc:
        logger.warning(f'[wandb] init failed, continuing without it: {exc}')
        return False
    wandb.define_metric('train/global_step')
    wandb.define_metric('train/*', step_metric='train/global_step')
    wandb.define_metric('epoch/*', step_metric='epoch/index')
    return True


def wandb_log(use_wandb, payload):
    """ Guarded wandb.log: logging backend errors must not kill a run """
    if not use_wandb or wandb is None or wandb.run is None:
        return
    try:
        wandb.log(payload)
    except Exception as exc:
        logger.warning(f'[wandb] log failed: {exc}')


class NoDDP(torch.nn.Module):
    """ Stand-in for DistributedDataParallel when no process group exists (CPU).

    Exposes `.module` and keeps the `module.` state_dict prefix, so checkpoints
    stay interchangeable with DDP runs.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
        # -- CPU (e.g. debugging): no bf16 autocast / cuda GradScaler
        if use_bfloat16:
            logger.info('No GPU available: disabling use_bfloat16')
        use_bfloat16 = False
    else:
        # -- one process per GPU: pick the local-rank GPU when the launcher
        # -- exposes the whole node (srun/torchrun), cuda:0 when it already
        # -- restricted CUDA_VISIBLE_DEVICES to a single device (main.py)
        device = torch.device(f'cuda:{get_local_rank() % torch.cuda.device_count()}')
        torch.cuda.set_device(device)

    # -- DATA
    dataset_name = args['data'].get('dataset', 'imagenet1k')
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    resize = args['data']['use_resizing_transformation']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem'] and device.type == 'cuda'
    num_workers = args['data']['num_workers']
    root_path = args['data'].get('root_path', None)
    image_folder = args['data'].get('image_folder', None)
    # -- (radimgnet only)
    data_paths = args['data'].get('data_paths', None)
    color_mode = args['data'].get('color_mode', 'RGB')
    drop_duplicates = args['data'].get('drop_duplicates', True)
    verify_exists = args['data'].get('verify_exists', False)
    max_retries = args['data'].get('max_retries', None)
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- FREQUENCY MASK (MFM-style filtering of the context-encoder input)
    use_freq_mask = (args.get('freq_mask', None) or {}).get('enable', False)
    # -- before_crop: the dataset filters the full image before cropping
    freq_before_crop = use_freq_mask and args['freq_mask'].get('before_crop', False)
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']
    save_every = args['optimization']['save_every']

    # -- LOGGING
    folder = os.path.join(args['wandb']['root'], args['wandb']['run_name'])
    tag = args['logging']['write_tag']

    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    logger.info(f'Per-GPU batch size {batch_size}, global batch size {batch_size * world_size}')

    os.makedirs(folder, exist_ok=True)
    if rank == 0:
        dump = os.path.join(folder, 'params-ijepa.yaml')
        with open(dump, 'w') as f:
            yaml.dump(args, f)
    # -- DDP needs an initialized (nccl) process group, i.e. a GPU
    use_ddp = (device.type == 'cuda') and dist.is_available() and dist.is_initialized()
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- init wandb (rank 0 only)
    use_wandb = init_wandb(args, rank, folder)

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    target_encoder = copy.deepcopy(encoder)

    # -- make data transforms
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)

    normalization = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = make_transforms(
        crop_size=crop_size,
        resize=resize,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter,
        normalization=normalization)

    # -- init data-loaders/samplers
    if use_freq_mask and dataset_name != 'radimgnet':
        raise NotImplementedError('freq_mask is only implemented for the radimgnet dataset')
    if dataset_name == 'imagenet1k':
        unsupervised_dataset, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
                transform=transform,
                batch_size=batch_size,
                collator=mask_collator,
                pin_mem=pin_mem,
                training=True,
                num_workers=num_workers,
                world_size=world_size,
                rank=rank,
                root_path=root_path,
                image_folder=image_folder,
                copy_data=copy_data,
                drop_last=True)
    elif dataset_name == 'radimgnet':
        unsupervised_dataset, unsupervised_loader, unsupervised_sampler = make_radimgnet(
                transform=transform,
                batch_size=batch_size,
                collator=mask_collator,
                pin_mem=pin_mem,
                num_workers=num_workers,
                world_size=world_size,
                rank=rank,
                data_paths=data_paths,
                root=root_path,
                color_mode=color_mode,
                drop_duplicates=drop_duplicates,
                verify_exists=verify_exists,
                max_retries=max_retries,
                drop_last=True,
                config=args,
                )
    else:
        raise ValueError(f'unknown dataset: {dataset_name}')
    ipe = len(unsupervised_loader)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    if use_ddp:
        encoder = DistributedDataParallel(encoder, device_ids=[device.index], static_graph=True)
        predictor = DistributedDataParallel(predictor, device_ids=[device.index], static_graph=True)
        target_encoder = DistributedDataParallel(target_encoder, device_ids=[device.index])
    else:
        logger.info('No GPU process group: running without DistributedDataParallel')
        encoder = NoDDP(encoder)
        predictor = NoDDP(predictor)
        target_encoder = NoDDP(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        if use_ddp:
            for _ in range(start_epoch*ipe):
                scheduler.step()
                wd_scheduler.step()
                next(momentum_scheduler)
                mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            # if (epoch) % checkpoint_freq == 0:
            torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)
        if hasattr(unsupervised_dataset, 'set_epoch'):
            unsupervised_dataset.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        save_checkpoint(-1)

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs():
                # -- unsupervised imgs
                imgs = udata[0].to(device, non_blocking=True)
                imgs_ctx = imgs.clone()
                if freq_before_crop:
                    # -- udata = [imgs, targets, imgs_ctx]; filtered in the dataset
                    # -- before the crop, then cropped like the clean image
                    imgs_ctx = udata[2].to(device, non_blocking=True)
                elif use_freq_mask:
                    # -- udata = [imgs, targets, freq_masks]; the context encoder
                    # -- sees the filtered image, the target encoder the clean one
                    fmasks = udata[2].to(device, non_blocking=True)
                    filtered = (fmasks != 1).flatten(1).any(1)  # (B,)
                    imgs_ctx[filtered] = apply_freq_mask(imgs[filtered], fmasks[filtered], *normalization)
                else:
                    imgs_ctx = imgs
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, imgs_ctx, masks_1, masks_2)
            imgs, imgs_ctx, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_target():
                    with torch.no_grad():
                        # -- Dim = bsz, npatches, embed_dim
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        # -- Dim = bsz * n#targets, n# patches per target, embed_dim
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc)) # Repeat is the number of context patches per sample
                        return h

                def forward_context():
                    z = encoder(imgs_ctx, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    return z # predicted target from the context embeddings

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target() # -- Dim = bsz*ntargets, n patches in target, embed_dim
                    z = forward_context()
                    loss = loss_fn(z, h)

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats)
            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    payload = {
                        'train/global_step': epoch * ipe + itr,
                        'train/epoch': epoch + 1,
                        'train/loss': loss,
                        'train/loss_avg': loss_meter.avg,
                        'train/lr': _new_lr,
                        'train/wd': _new_wd,
                        'train/mask_enc': maskA_meter.val,
                        'train/mask_pred': maskB_meter.val,
                        'train/time_ms': etime,
                        'train/mem_mb': torch.cuda.max_memory_allocated() / 1024.**2}
                    if grad_stats is not None:
                        payload.update({
                            'train/grad_first_layer': grad_stats.first_layer,
                            'train/grad_last_layer': grad_stats.last_layer,
                            'train/grad_min': grad_stats.min,
                            'train/grad_max': grad_stats.max})
                    wandb_log(use_wandb, payload)
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'
        if (epoch % save_every == 0) or (epoch == num_epochs - 1):
            # -- Save Checkpoint after every X epochs
            logger.info('avg. loss %.3f' % loss_meter.avg)
            wandb_log(use_wandb, {
                'epoch/index': epoch,
                'epoch/loss': loss_meter.avg,
                'epoch/time_ms': time_meter.avg})
            save_checkpoint(epoch)

    if use_wandb:
        wandb.finish()
    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    # -- single-process entry point (debugging): python -m src.train --fname cfg.yaml
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, help='name of config file to load')
    parser.add_argument('--port', type=int, default=40112,
                        help='rendezvous port for torch.distributed')
    cli_args = parser.parse_args()
    params = load_config(cli_args.fname)
    # -- DDP wrappers need a (nccl) process group, even with a single process;
    # -- on CPU main() falls back to NoDDP instead
    if torch.cuda.is_available():
        init_distributed(port=cli_args.port, rank_and_world_size=(0, 1))
    main(args=params)
