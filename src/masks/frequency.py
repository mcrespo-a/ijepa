#
# Frequency-domain masking, ported from MFM (data/data_mfm.py, models/mfm.py).
#
# Mask convention (same as MFM): masks are [H, W] in the *centred* spectrum
# (low frequencies in the middle, i.e. after fftshift), 1 = frequency kept,
# 0 = frequency removed.
#
# Masks are sampled per image in the dataset (CPU, data-loader workers, see
# src/datasets/radimgnet.py), resampled onto the full-resolution image with
# `resample_freq_mask` and applied with `filter_image` before the crop, on the
# raw image in [0, 1], as MFM filters the raw image before normalising it.
#

from logging import getLogger

import numpy as np
import torch

logger = getLogger()


class MFMMaskGenerator:
    """ MFM default: random low-pass or high-pass filter with a circular cutoff """

    needs_image = False

    def __init__(self, input_size=224, mask_radius1=16, mask_radius2=999, sample_ratio=0.5):
        self.input_size = input_size
        self.sample_ratio = sample_ratio
        c = input_size // 2
        yy, xx = np.ogrid[:input_size, :input_size]
        d2 = (xx - c) ** 2 + (yy - c) ** 2
        band = (d2 >= mask_radius1 ** 2) & (d2 < mask_radius2 ** 2)
        self.mask = torch.from_numpy(~band)  # low-pass: keep the centre

    def __call__(self, img=None):
        if torch.rand(1).item() < self.sample_ratio:
            return self.mask.clone()  # low-pass
        return ~self.mask  # high-pass


class LowPassMaskGenerator:
    """ MFM low-pass / high-pass filter with a random circular cutoff: the
    radius is drawn uniformly from [radius_min, radius_max] for every image """

    needs_image = False

    def __init__(self, input_size=224, radius_min=16, radius_max=None, sample_ratio=0.5):
        self.input_size = input_size
        self.sample_ratio = sample_ratio
        c = input_size // 2
        # default upper bound: Nyquist (largest circle inside the spectrum)
        self.radius_min = radius_min
        self.radius_max = c if radius_max is None else radius_max
        assert 0 <= self.radius_min <= self.radius_max, \
            f'invalid cutoff range [{self.radius_min}, {self.radius_max}]'
        yy, xx = np.ogrid[:input_size, :input_size]
        self.d2 = torch.from_numpy((xx - c) ** 2 + (yy - c) ** 2)

    def __call__(self, img=None):
        r = torch.randint(self.radius_min, self.radius_max + 1, (1,)).item()
        mask = self.d2 < r ** 2  # low-pass: keep the centre
        if torch.rand(1).item() < self.sample_ratio:
            return mask  # low-pass
        return ~mask  # high-pass


class FrequencyUSFM:
    """ USFM band-stop: remove random annuli until the kept area < kept_area """

    needs_image = False

    def __init__(self, input_size=224, center_protect=3, base_bw=4,
                 increase_factor=0.5, max_increase=16, kept_area=0.65):
        self.input_size = input_size
        self.kept_area = kept_area
        n = input_size
        c = n // 2
        yy, xx = np.ogrid[:n, :n]
        dist = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)

        f_mask_group = []
        start, end, bw = c - center_protect, c + center_protect, base_bw
        while start >= 0:
            r = end - c
            rap = np.ones((n, n), dtype=np.float32)
            rap[(dist >= r - bw) & (dist <= r)] = 0.0  # stop annulus [r-bw, r]
            f_mask_group.append(torch.from_numpy(rap))
            start, end = c - bw, c + bw
            bw = bw + min(int(bw * increase_factor), max_increase)

        rap_mask = [(f_mask_group[i + 1] - f_mask_group[i] + 1).clamp(0, 1)
                    for i in range(len(f_mask_group) - 1)]
        self.f_mask_group = torch.stack(rap_mask, dim=0).bool()

    def __call__(self, img=None):
        m = torch.ones_like(self.f_mask_group[0])
        for i in torch.randperm(len(self.f_mask_group)).tolist():
            m &= self.f_mask_group[i]
            if m.float().mean() < self.kept_area:
                break
        return m


class FrequencyComRCom:
    """ FOLK data-adaptive Com / RCom mask: keep the top (Com) or bottom (RCom)
    `rate` fraction of the image's own spectral magnitudes """

    needs_image = True

    def __init__(self, input_size=224, rate_min=0.05, rate_max=0.95, com_prob=0.5):
        self.input_size = input_size
        self.rate_min = rate_min
        self.rate_max = rate_max
        self.com_prob = com_prob

    @torch.no_grad()
    def __call__(self, img):
        """ img: [C, H, W] -> bool mask [H, W] (centred), 1 = kept """
        h, w = img.shape[-2:]
        n = h * w
        spec = torch.fft.fft2(img.float().mean(0)).abs()
        # make |F(u,v)| and |F(-u,-v)| identical so the mask stays Hermitian
        spec = 0.5 * (spec + torch.roll(torch.flip(spec, (-2, -1)), (1, 1), (-2, -1)))
        flat = spec.reshape(n)

        rate = torch.empty(1).uniform_(self.rate_min, self.rate_max).item()
        is_com = torch.rand(1).item() < self.com_prob
        k = n - min(int(round(rate * n)), n)
        thr = flat.kthvalue(max(k, 1)).values
        com = flat > thr
        mask = com if is_com else ~com
        return torch.fft.fftshift(mask.reshape(h, w), dim=(-2, -1))


def make_freq_mask_generator(cfg, crop_size):
    """ Build a generator from the `freq_mask` config section, or None if disabled """
    cfg = cfg or {}
    if not cfg.get('enable', False):
        return None
    name = cfg.get('name', 'default')
    if name == 'default':
        gen = MFMMaskGenerator(
            input_size=crop_size,
            mask_radius1=cfg.get('mask_radius1', 16),
            mask_radius2=cfg.get('mask_radius2', 999),
            sample_ratio=cfg.get('sample_ratio', 0.5))
    elif name == 'LF':
        lf = cfg.get('lf', None) or {}
        gen = LowPassMaskGenerator(
            input_size=crop_size,
            radius_min=lf.get('radius_min', 16),
            radius_max=lf.get('radius_max', None),
            sample_ratio=cfg.get('sample_ratio', 0.5))
    elif name == 'usfm':
        usfm = cfg.get('usfm', None) or {}
        gen = FrequencyUSFM(
            input_size=crop_size,
            center_protect=usfm.get('center_protect', 3),
            base_bw=usfm.get('base_bw', 4),
            increase_factor=usfm.get('increase_factor', 0.5),
            max_increase=usfm.get('max_increase', 16),
            kept_area=usfm.get('kept_area', 0.65))
    elif name == 'folk':
        gen = FrequencyComRCom(
            input_size=crop_size,
            rate_min=cfg.get('rate_min', 0.05),
            rate_max=cfg.get('rate_max', 0.95),
            com_prob=cfg.get('com_prob', 0.5))
    else:
        raise ValueError(f'unknown freq_mask.name: {name}')
    logger.info(f'frequency masking enabled: {name} ({type(gen).__name__})')
    return gen


def resample_freq_mask(mask, out_hw):
    """ Resample a centred [h, w] mask onto a centred grid of size out_hw.

    Frequencies are matched in units of each grid's Nyquist frequency, so a
    mask built for the crop size removes the same relative band of a larger
    (full-resolution) image.
    """
    mask = torch.as_tensor(mask)

    def index(n_out, n_src):
        nu = (torch.arange(n_out) - n_out // 2) / (n_out / 2)
        idx = torch.round(nu * (n_src / 2)).long() + n_src // 2
        return idx.clamp(0, n_src - 1)

    return mask[index(out_hw[0], mask.shape[0])][:, index(out_hw[1], mask.shape[1])]


@torch.no_grad()
def filter_image(x, mask):
    """ Filter one raw [C, H, W] image in [0, 1] with a centred [H, W] mask, 1 = kept """
    x_freq = torch.fft.fftshift(torch.fft.fft2(x.float()), dim=(-2, -1))
    x_freq = x_freq * mask.to(x_freq.real.dtype)
    x = torch.fft.ifft2(torch.fft.ifftshift(x_freq, dim=(-2, -1))).real
    return torch.clamp(x, min=0., max=1.)
