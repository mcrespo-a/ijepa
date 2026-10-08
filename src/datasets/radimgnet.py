#
# RadImageNet dataset indexed by CSV or NPY manifests.
#
# Ported from MFM (data/radimgnet/create_dataset.py). Only the dataset
# indexing / loading logic is reused; the MFM-specific transforms and collate
# function are not. Transforms and the multi-block mask collator are the
# standard I-JEPA ones (src/transforms.py, src/masks/multiblock.py).
#
# `__getitem__` returns `(img, target)`, the same contract as the ImageNet
# loader, so `MaskCollator` yields `(udata, masks_enc, masks_pred)` with
# `udata = [imgs, targets]`.
#

import os
import zlib

from logging import getLogger
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = getLogger()

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')

PATH_COLUMN_CANDIDATES = (
    'path', 'filepath', 'file_path', 'fullpath', 'full_path',
    'image', 'image_path', 'img_path', 'img', 'filename', 'file_name',
    'file', 'fname', 'imagepath',
)

LABEL_COLUMN_CANDIDATES = ('label', 'class', 'target', 'y', 'diagnosis', 'organ')


def make_radimgnet(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    data_paths=None,
    root=None,
    color_mode='RGB',
    drop_duplicates=True,
    verify_exists=False,
    max_retries=None,
    drop_last=True
):
    dataset = RadImgNet(
        data_paths=data_paths,
        transform=transform,
        root=root,
        mode=color_mode,
        drop_duplicates=drop_duplicates,
        verify_exists=verify_exists,
        max_retries=max_retries)
    logger.info('RadImgNet dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('RadImgNet unsupervised data loader created')

    return dataset, data_loader, dist_sampler


def _looks_like_image_path(value) -> bool:
    return isinstance(value, str) and value.strip().strip('\'"').lower().endswith(IMAGE_EXTENSIONS)


def _read_csv_manifest(data_path: str) -> Tuple[List[str], Optional[List[str]]]:
    """Read one CSV manifest, inferring delimiter, header presence and columns.

    Header presence is sniffed from the first physical line: if its first field
    already terminates in an image extension, the file is head-less.
    """
    with open(data_path, 'r') as fh:
        first_line = fh.readline().rstrip('\n').rstrip('\r')
    if not first_line:
        raise ValueError(f'{data_path}: empty manifest')

    sep = '::' if ('::' in first_line and ',' not in first_line) else ','
    engine = 'python' if len(sep) > 1 else 'c'
    has_header = not _looks_like_image_path(first_line.split(sep)[0])

    frame = pd.read_csv(
        data_path,
        sep=sep,
        engine=engine,
        header=0 if has_header else None,
        dtype=str,
        keep_default_na=False,
        skip_blank_lines=True,
    )
    if frame.shape[1] == 0 or len(frame) == 0:
        raise ValueError(f'{data_path}: no usable rows')

    if has_header:
        normalised = {str(c).strip().lower(): c for c in frame.columns}
        path_col = next((normalised[c] for c in PATH_COLUMN_CANDIDATES if c in normalised), None)
        label_col = next((normalised[c] for c in LABEL_COLUMN_CANDIDATES if c in normalised), None)
        if path_col is None:
            # Unrecognised header: fall back to positional convention.
            path_col = frame.columns[0]
            label_col = frame.columns[1] if frame.shape[1] > 1 else None
            logger.warning(
                '%s: unrecognised header %s, using column 0 as path',
                os.path.basename(data_path), list(frame.columns),
            )
    else:
        path_col = frame.columns[0]
        label_col = frame.columns[1] if frame.shape[1] > 1 else None

    paths = frame[path_col].astype(str).str.strip().str.strip('\'"').tolist()
    labels = frame[label_col].astype(str).str.strip().tolist() if label_col is not None else None
    return paths, labels


def _read_npy_manifest(data_path: str) -> Tuple[List[str], Optional[List[str]]]:
    """Read a .npy array of paths. Handles str, np.str_ and bytes elements."""
    data = np.load(data_path, allow_pickle=True)
    paths = []
    for item in np.asarray(data).ravel().tolist():
        if isinstance(item, bytes):
            item = item.decode('utf-8')
        paths.append(str(item).strip().strip('\'"'))
    return paths, None


class RadImgNet(Dataset):

    def __init__(
        self,
        data_paths,
        transform=None,
        root=None,
        mode='RGB',
        drop_duplicates=True,
        verify_exists=False,
        max_retries=None
    ):
        """
        RadImgNet

        One sample = one radiologic image listed in a manifest.

        :param data_paths: manifest path or list of manifest paths (.csv or .npy)
        :param transform: callable applied to the PIL image
        :param root: prefix prepended to relative paths found in the manifests
        :param mode: PIL colour mode requested before the transform ('RGB'
            replicates single-channel images threefold, as needed by in_chans=3)
        :param drop_duplicates: remove repeated paths across manifests
        :param verify_exists: stat every row at construction (expensive on
            networked storage; enable for a first run only)
        :param max_retries: substitution budget per item when a file is
            unreadable (defaults to len(dataset))
        """
        super().__init__()
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        # Tolerate stray quoting / trailing commas surviving from YAML editing.
        data_paths = [str(p).strip().rstrip(',').strip('\'"') for p in (data_paths or [])]
        if len(data_paths) == 0:
            raise ValueError('RadImgNet: no manifests supplied (data.data_paths is empty)')

        samples, labels = [], []
        any_label = False

        for data_path in data_paths:
            if not os.path.isfile(data_path):
                raise FileNotFoundError(f'manifest not found: {data_path}')
            if data_path.endswith('.csv'):
                paths, labs = _read_csv_manifest(data_path)
            elif data_path.endswith('.npy'):
                paths, labs = _read_npy_manifest(data_path)
            else:
                raise ValueError(f'unsupported manifest extension: {data_path}')

            if labs is None:
                labs = [''] * len(paths)
            else:
                any_label = True
            samples.extend(paths)
            labels.extend(labs)
            logger.info(f'  {os.path.basename(data_path)}: {len(paths)} rows')

        if root:
            samples = [p if os.path.isabs(p) else os.path.join(root, p) for p in samples]

        n_raw = len(samples)

        if drop_duplicates:
            seen = set()
            keep = [i for i, p in enumerate(samples) if not (p in seen or seen.add(p))]
            samples = [samples[i] for i in keep]
            labels = [labels[i] for i in keep]

        if verify_exists:
            exists = [os.path.isfile(p) for p in samples]
            missing = [p for p, ok in zip(samples, exists) if not ok]
            if missing:
                logger.warning(f'{len(missing)} listed files do not exist and were dropped')
                for p in missing[:5]:
                    logger.warning(f'  missing: {p}')
            samples = [p for p, ok in zip(samples, exists) if ok]
            labels = [v for v, ok in zip(labels, exists) if ok]

        if len(samples) == 0:
            raise RuntimeError('RadImgNet is empty after indexing')

        self.samples = samples
        self.transform = transform
        self.mode = mode
        self.max_retries = max_retries if max_retries is not None else len(samples)
        # Plain int (not shared memory): I-JEPA uses the 'spawn' start method
        # and non-persistent workers, so workers are re-created every epoch and
        # pick up the value set by `set_epoch`.
        self._epoch = 0

        # Integer targets. Unused by the I-JEPA objective, but the collator
        # requires a collatable type.
        if any_label:
            self.classes = sorted({v for v in labels if v != ''})
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            self.targets = [self.class_to_idx.get(v, -1) for v in labels]
        else:
            self.classes = []
            self.class_to_idx = {}
            self.targets = [0] * len(samples)

        logger.info(
            f'RadImgNet: {len(data_paths)} manifests, {n_raw} rows, '
            f'{len(self.samples)} unique images, {len(self.classes)} classes')

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _sample_key(fname):
        return zlib.crc32(fname.encode('utf-8')) & 0xFFFFFFFF  # stable; NOT builtin hash()

    def set_epoch(self, epoch):
        """ Advance the substitution stream. Call alongside `sampler.set_epoch` """
        self._epoch = int(epoch)

    def _load_image(self, path):
        """ Return a PIL image, or None if the file is unreadable """
        try:
            with open(path, 'rb') as fh:
                img = Image.open(fh)
                img.load()
            return img.convert(self.mode) if img.mode != self.mode else img
        except Exception as exc:  # noqa: BLE001
            logger.warning(f'unreadable image {path} ({type(exc).__name__})')
            return None

    def __getitem__(self, index):
        anchor = self._sample_key(self.samples[index])  # anchor the stream to the failing slot
        retry = 0
        while True:
            img = self._load_image(self.samples[index])
            if img is not None:
                break
            retry += 1
            if retry > self.max_retries:
                raise RuntimeError('no loadable image found in dataset')
            # Uniform over the dataset, but a pure function of
            # (slot, epoch, retry), hence reproducible across runs and ranks.
            rng = np.random.default_rng((anchor, self._epoch, 0x5117, retry))
            index = int(rng.integers(0, len(self)))

        target = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target
