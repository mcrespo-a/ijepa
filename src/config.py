#
# Default configuration, following MFM's config.py: every option has a default
# here, and a YAML file only declares the values it customises.
#
# YAML files may inherit from other YAML files with
#     BASE: ['pretrain.yaml']
# (paths relative to the file itself). Bases are merged first, in order, then
# the file's own values. Unknown keys raise, so typos do not pass silently.
#
# The result is a plain nested dict, so `args['data']['batch_size']` works as
# in the original I-JEPA code.
#

import copy
import os
from logging import getLogger

import yaml

logger = getLogger()

_C = {
    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    'data': {
        'dataset': 'imagenet1k',         # imagenet1k | radimgnet
        'batch_size': 256,               # per GPU
        'num_workers': 5,
        'pin_mem': True,
        'crop_size': 224,
        'crop_scale': [0.3, 1.0],
        'use_resizing_transformation': False,
        'use_color_distortion': False,
        'color_jitter_strength': 0.0,
        'use_gaussian_blur': False,
        'use_horizontal_flip': False,
        'root_path': None,               # prefix for relative paths (radimgnet) / dataset root (imagenet1k)
        # -- imagenet1k only
        'image_folder': None,
        # -- radimgnet only
        'data_paths': [],                # CSV / NPY manifests
        'color_mode': 'RGB',
        'drop_duplicates': True,
        'verify_exists': False,          # stat every file at start-up (slow on network storage)
        'max_retries': None,             # None = len(dataset)
    },
    # -------------------------------------------------------------------------
    # Multi-block spatial masks
    # -------------------------------------------------------------------------
    'mask': {
        'patch_size': 16,
        'allow_overlap': False,          # overlap between context and target blocks
        'num_enc_masks': 1,              # context blocks
        'enc_mask_scale': [0.85, 1.0],
        'min_keep': 10,                  # min patches in a context block
        'num_pred_masks': 4,             # target blocks
        'pred_mask_scale': [0.15, 0.2],
        'aspect_ratio': [0.75, 1.5],
    },
    # -------------------------------------------------------------------------
    # Frequency mask (MFM defaults, see MFM config.py _C.DATA.*)
    # -------------------------------------------------------------------------
    'freq_mask': {
        'enable': False,
        'name': 'default',               # default (low/high-pass) | usfm (band-stop annuli) | folk (Com/RCom)
        'prob': 0.0,                     # probability of filtering each image
        # -- default
        'mask_radius1': 16,
        'mask_radius2': 999,
        'sample_ratio': 1.0,             # P(low-pass); high-pass otherwise
        # -- usfm
        'usfm': {
            'center_protect': 3,
            'base_bw': 4,
            'increase_factor': 0.5,
            'max_increase': 16,
            'kept_area': 0.65,
        },
        # -- folk
        'rate_min': 0.05,
        'rate_max': 0.95,
        'com_prob': 0.5,
    },
    # -------------------------------------------------------------------------
    # Model / checkpointing
    # -------------------------------------------------------------------------
    'meta': {
        'model_name': 'vit_small',
        'pred_depth': 12,
        'pred_emb_dim': 384,
        'use_bfloat16': True,
        'load_checkpoint': False,
        'read_checkpoint': None,         # file name inside the run folder; None = latest
        'copy_data': False,
    },
    # -------------------------------------------------------------------------
    # Optimisation
    # -------------------------------------------------------------------------
    'optimization': {
        'epochs': 300,
        'save_every': 50,
        'warmup': 40,
        'start_lr': 0.0002,
        'lr': 0.001,
        'final_lr': 1.0e-06,
        'weight_decay': 0.04,
        'final_weight_decay': 0.4,
        'ema': [0.996, 1.0],
        'ipe_scale': 1.0,
    },
    # -------------------------------------------------------------------------
    # Logging (run folder = wandb.root / wandb.run_name)
    # -------------------------------------------------------------------------
    'logging': {
        'folder': None,
        'write_tag': 'jepa',
    },
    'wandb': {
        'enable': False,
        'root': './experiments',
        'run_name': 'default',
        'mode': 'offline',
        'project': 'ijepa-pretrain',
        'entity': None,
        'group': None,
        'run_id': None,
        'dir': None,
    },
}


def _merge(base, override, prefix=''):
    """ Recursively merge `override` into `base` in place; unknown keys raise """
    for k, v in override.items():
        name = f'{prefix}{k}'
        if k not in base:
            raise KeyError(f'unknown config key: {name}')
        if isinstance(base[k], dict):
            if not isinstance(v, dict):
                raise TypeError(f'config key {name} must be a mapping, got {v!r}')
            _merge(base[k], v, prefix=f'{name}.')
        else:
            base[k] = v


def _merge_from_file(config, fname):
    with open(fname, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader) or {}
    for base in cfg.pop('BASE', None) or []:
        if base:
            _merge_from_file(config, os.path.join(os.path.dirname(fname), base))
    logger.info(f'=> merge config from {fname}')
    _merge(config, cfg)


def get_default_config():
    return copy.deepcopy(_C)


def load_config(fname):
    """ Defaults, then BASE files, then `fname` """
    config = get_default_config()
    _merge_from_file(config, fname)
    return config
