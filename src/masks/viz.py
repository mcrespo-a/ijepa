#
# Debug visualisation of the multiblock masks (notebooks/multiblock_masks.ipynb
# as a function). Meant to be called from a debugger paused in src/train.py,
# after `imgs, masks_enc, masks_pred = load_imgs()`:
#
#   from src.masks.viz import plot_masks
#   plot_masks(imgs, masks_enc, masks_pred, patch_size=patch_size,
#              out=f'{folder}/mask_viz/ep{epoch}_itr{itr}.png')
#
# Renders with the Agg canvas (no pyplot), so it works headless on a compute
# node and does not touch the debugger's matplotlib backend.
#

import os

import numpy as np
import torch

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Patch
from matplotlib import colormaps

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CONTEXT_RGB = (0.55, 0.75, 0.95)
OVERLAP_RGB = (0., 0., 0.)


def _idx_to_grid(idx, gh, gw):
    """ Flat patch indices -> [gh, gw] binary grid """
    g = np.zeros(gh * gw, dtype=int)
    g[np.asarray(idx.detach().cpu() if torch.is_tensor(idx) else idx, dtype=int)] = 1
    return g.reshape(gh, gw)


def _draw_grid(ax, gh, gw):
    ax.set_xticks(np.arange(-.5, gw, 1), minor=True)
    ax.set_yticks(np.arange(-.5, gh, 1), minor=True)
    ax.grid(which='minor', color='k', linewidth=0.3, alpha=0.3)
    ax.tick_params(which='both', length=0)
    ax.set_xticks([]); ax.set_yticks([])


def mask_stats(masks_enc, masks_pred, grid_size):
    """ Per-image patch counts and overlaps of one collated batch.

    :param masks_enc: list (nenc) of [B, K_enc] index tensors
    :param masks_pred: list (npred) of [B, K_pred] index tensors
    :param grid_size: (gh, gw) patch grid
    :returns: dict of np arrays of length B
    """
    gh, gw = grid_size
    N = gh * gw
    B = len(masks_enc[0])
    rows = []
    for b in range(B):
        enc = set().union(*[set(m[b].tolist()) for m in masks_enc])
        preds = [set(m[b].tolist()) for m in masks_pred]
        tgt = set().union(*preds)
        rows.append(dict(
            K_enc=len(masks_enc[0][b]),
            K_pred=len(masks_pred[0][b]),
            enc_frac=len(enc) / N,
            tgt_frac=len(tgt) / N,
            enc_tgt_overlap=len(enc & tgt),
            pred_pred_overlap=sum(len(preds[i] & preds[j])
                                  for i in range(len(preds)) for j in range(i + 1, len(preds))),
            unused_frac=1 - len(enc | tgt) / N))
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def describe_masks(masks_enc, masks_pred, grid_size):
    """ Printable summary: shapes, then per-image stats """
    s = mask_stats(masks_enc, masks_pred, grid_size)
    gh, gw = grid_size
    lines = [
        f'grid {gh}x{gw} = {gh * gw} patches',
        f'masks_enc : {len(masks_enc)} x {tuple(masks_enc[0].shape)}  (patches the context encoder keeps)',
        f'masks_pred: {len(masks_pred)} x {tuple(masks_pred[0].shape)}  (patches of each target block)',
        f'{"img":>3} | {"K_enc":>5} {"K_pred":>6} | {"ctx%":>5} {"tgt%":>5} {"unused%":>7} | ctx∩tgt tgt∩tgt',
    ]
    for b in range(len(s['K_enc'])):
        lines.append(
            f'{b:>3} | {s["K_enc"][b]:>5} {s["K_pred"][b]:>6} | '
            f'{100 * s["enc_frac"][b]:>5.1f} {100 * s["tgt_frac"][b]:>5.1f} {100 * s["unused_frac"][b]:>7.1f} | '
            f'{s["enc_tgt_overlap"][b]:>7} {s["pred_pred_overlap"][b]:>7}')
    return '\n'.join(lines)


def plot_masks(
    imgs,
    masks_enc,
    masks_pred,
    patch_size=16,
    out='mask_viz.png',
    max_images=4,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    verbose=True,
):
    """ Save a figure of the context / target masks of one collated batch.

    One row per image: the image with the patches hidden from the encoder
    darkened and the target blocks outlined, each context mask, each target
    mask, and an overlay of all of them on the patch grid.

    :param imgs: [B, C, H, W] normalised batch (any device)
    :param masks_enc: list (nenc) of [B, K_enc] index tensors
    :param masks_pred: list (npred) of [B, K_pred] index tensors
    :param patch_size: model patch size (config mask.patch_size)
    :param out: PNG path; parent directories are created
    :param max_images: number of images (rows) to draw
    :param mean, std: normalisation to undo for display (None: no undo)
    :param verbose: print describe_masks() and the saved path
    :returns: out
    """
    imgs = imgs.detach().float().cpu()
    B, C, H, W = imgs.shape
    gh, gw = H // patch_size, W // patch_size
    nenc, npred = len(masks_enc), len(masks_pred)
    nb = min(B, max_images)

    if mean is not None and std is not None and C == len(mean):
        imgs = imgs * torch.tensor(std).view(1, -1, 1, 1) + torch.tensor(mean).view(1, -1, 1, 1)
    imgs = imgs.clamp(0, 1).permute(0, 2, 3, 1).numpy()
    if C == 1:
        imgs = np.repeat(imgs, 3, axis=-1)

    cmap_t = colormaps['tab10']
    tgt_color = [cmap_t(1 + p)[:3] for p in range(npred)]
    up = np.ones((patch_size, patch_size))

    ncols = 1 + nenc + npred + 1
    fig = Figure(figsize=(2.2 * ncols, 2.3 * nb))
    FigureCanvasAgg(fig)
    axes = np.asarray(fig.subplots(nb, ncols, squeeze=False))

    for b in range(nb):
        enc_grids = [_idx_to_grid(m[b], gh, gw) for m in masks_enc]
        tgt_grids = [_idx_to_grid(m[b], gh, gw) for m in masks_pred]
        seen = np.clip(sum(enc_grids), 0, 1)

        # -- image: darken what the context encoder drops, outline targets
        ax = axes[b, 0]
        ax.imshow(imgs[b])
        hidden = np.kron(1 - seen, up)
        ax.imshow(np.ma.masked_where(hidden == 0, hidden), cmap='gray_r', alpha=0.6, vmin=0, vmax=1)
        for p, g in enumerate(tgt_grids):
            ax.contour(np.kron(g, up), levels=[0.5], colors=[tgt_color[p]], linewidths=1.5)
        ax.set_title(f'img {b}: dark = not in context', fontsize=8)
        ax.axis('off')

        # -- each context mask
        for e, g in enumerate(enc_grids):
            ax = axes[b, 1 + e]
            ax.imshow(g, cmap='Blues', vmin=0, vmax=1.3)
            ax.set_title(f'enc {e} ({int(g.sum())})', fontsize=8)
            _draw_grid(ax, gh, gw)

        # -- each target mask
        for p, g in enumerate(tgt_grids):
            ax = axes[b, 1 + nenc + p]
            ax.imshow(g, cmap='Oranges', vmin=0, vmax=1.3)
            ax.set_title(f'pred {p} ({int(g.sum())})', fontsize=8)
            _draw_grid(ax, gh, gw)

        # -- overlay; context ∩ target (overlap leak) in black
        ax = axes[b, -1]
        rgb = np.ones((gh, gw, 3))
        rgb[seen == 1] = CONTEXT_RGB
        for p, g in enumerate(tgt_grids):
            rgb[g == 1] = tgt_color[p]
        rgb[(seen == 1) & (np.clip(sum(tgt_grids), 0, 1) == 1)] = OVERLAP_RGB
        ax.imshow(rgb)
        ax.set_title('overlay', fontsize=8)
        _draw_grid(ax, gh, gw)

    fig.legend(handles=[Patch(color=CONTEXT_RGB, label='context (enc)')]
               + [Patch(color=tgt_color[p], label=f'target {p}') for p in range(npred)]
               + [Patch(color=OVERLAP_RGB, label='context ∩ target')],
               loc='lower center', ncol=npred + 2, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=110)

    if verbose:
        print(describe_masks(masks_enc, masks_pred, (gh, gw)))
        print(f'saved {out}')
    return out
