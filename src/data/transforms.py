import albumentations as A
import cv2
import math
import numpy as np
import random
import scipy.stats as ss
from albumentations.random_utils import beta
from copy import deepcopy
from justpyplot import justpyplot as jplt
from scipy.signal import butter, lfilter
from typing import Dict, List, Callable, Union, Sequence, Iterator, Any, Literal
from warnings import warn

from .constants import (
    SPECTROGRAM_N_SAMPLES, 
    EED_N_SAMPLES, 
    EED_SAMPLING_RATE_HZ, 
    LABEL_COLS_ORDERED,
    N_SPECTROGRAM_TILES,
    MEL_N_FFT,
    EEG_DIFF_COL_INDICES,
    MEL_HOP_LENGTH,
    EEG_LR_FLIP_REORDER_INDICES,
    EEG_SPECTROGRAM_LR_FLIP_REORDER_INDICES,
    EEG_DIFF_ABS_MAX,
)


###################################################################
##################### CV ##########################################
###################################################################

class CopyPastePositive:
    """Copy masked area from one image to another.

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(
        self, 
        mask_index: int = 2,
        always_apply=True,
        p=1.0, 
    ):
        self.mask_index = mask_index
        self.always_apply = always_apply
        self.p = p

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, np.ndarray]:
        # Toss a coin
        if not force_apply and not self.always_apply and random.random() > self.p:
            return kwargs

        mask = (kwargs['masks'][self.mask_index] > 0) & (kwargs['masks1'][self.mask_index] <= 0)

        kwargs['image'][mask] = kwargs['image1'][mask]
        for i in range(len(kwargs['masks'])):
            kwargs['masks'][i][mask] = kwargs['masks1'][i][mask]

        return kwargs


# https://github.com/albumentations-team/albumentations/pull/1409/files
class MixUp:
    def __init__(
        self,
        alpha = 32.,
        beta = 32.,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        self.alpha = alpha
        self.beta = beta
        self.always_apply = always_apply
        self.p = p

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, np.ndarray]:
        # Toss a coin
        if not force_apply and not self.always_apply and random.random() > self.p:
            return kwargs

        h1, w1, _ = kwargs['image'].shape
        h2, w2, _ = kwargs['image1'].shape
        if h1 != h2 or w1 != w2:
            raise ValueError("MixUp transformation expects both images to have identical shape.")
        
        r = np.random.beta(self.alpha, self.beta)
        
        kwargs['image'] = (kwargs['image'] * r + kwargs['image1'] * (1 - r))
        for i in range(len(kwargs['masks'])):
            kwargs['masks'][i] = (kwargs['masks'][i] * r + kwargs['masks1'][i] * (1 - r))
        
        return kwargs


class CutMix:
    def __init__(
        self,
        width: int = 64,
        height: int = 64,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        self.width = width
        self.height = height
        self.always_apply = always_apply
        self.p = p

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, np.ndarray]:
        # Toss a coin
        if not force_apply and not self.always_apply and random.random() > self.p:
            return kwargs

        h, w, _ = kwargs['image'].shape
        h1, w1, _ = kwargs['image1'].shape
        if (
            h < self.height or 
            w < self.width or 
            h1 < self.height or 
            w1 < self.width
        ):
            raise ValueError("CutMix transformation expects both images to be at least {}x{} pixels.".format(self.max_height, self.max_width))

        # Get random bbox
        h_start = random.randint(0, h - self.height)
        w_start = random.randint(0, w - self.width)
        h_end = h_start + self.height
        w_end = w_start + self.width

        # Copy image and masks region
        kwargs['image'][h_start:h_end, w_start:w_end] = kwargs['image1'][h_start:h_end, w_start:w_end]
        for i in range(len(kwargs['masks'])):
            kwargs['masks'][i][h_start:h_end, w_start:w_end] = kwargs['masks1'][i][h_start:h_end, w_start:w_end]
        
        return kwargs


###################################################################
##################### HMS #########################################
###################################################################
class Subrecord:
    def _select_by_subrecord(self, subrecord, **item):
        # Extract sub EEG & spectrogram
        eeg = item['eeg']
        if eeg.ndim == 2:
            eeg_start_index = int(subrecord['eeg_label_offset_seconds'] * EED_SAMPLING_RATE_HZ)
            eeg_stop_index = eeg_start_index + EED_N_SAMPLES
            eeg = eeg[eeg_start_index:eeg_stop_index]
        else:
            assert eeg.ndim == 3
            eeg_start_index = int(subrecord['eeg_label_offset_seconds'] * EED_SAMPLING_RATE_HZ / MEL_N_FFT)
            eeg_stop_index = eeg_start_index + EED_N_SAMPLES // MEL_N_FFT
            eeg = eeg[eeg_start_index:eeg_stop_index]

        spectrogram = item['spectrogram']
        spectrogram_time = item['spectrogram_time']
        spectrogram_label_offset_seconds = subrecord['spectrogram_label_offset_seconds']
        spectrogram = spectrogram[spectrogram_time >= spectrogram_label_offset_seconds][:SPECTROGRAM_N_SAMPLES]
        spectrogram_time = spectrogram_time[spectrogram_time >= spectrogram_label_offset_seconds][:SPECTROGRAM_N_SAMPLES]

        eeg_spectrogram = item['eeg_spectrogram']
        if eeg_spectrogram is not None:
            eeg_spectrogram_start_index = eeg_start_index // MEL_HOP_LENGTH
            spectrogram_for_50_sec_len = (EED_N_SAMPLES - MEL_N_FFT) // MEL_HOP_LENGTH + 1
            eeg_spectrogram_stop_index = eeg_spectrogram_start_index + spectrogram_for_50_sec_len
            eeg_spectrogram = eeg_spectrogram[:, eeg_spectrogram_start_index:eeg_spectrogram_stop_index]
            eeg_spectrogram = eeg_spectrogram.astype(np.float32) / 255.0

        # Put back to item
        item['eeg'] = eeg
        item['eeg_spectrogram'] = eeg_spectrogram
        item['spectrogram'] = spectrogram
        item['spectrogram_time'] = spectrogram_time
        if 'label' in item and item['label'] is not None:
            item['label'] = subrecord[LABEL_COLS_ORDERED].values
        item['meta'] = subrecord

        return item


class RandomSubrecord(Subrecord):
    def __init__(self, mode='discrete'):
        self.mode = mode
    
    def __call__(self, *args, force_apply: bool = False, **item):
        if self.mode == 'discrete':
            # Sample single sub-record
            subrecord = item['meta'].sample().squeeze()
        elif self.mode == 'gauss_discrete':
            # https://stackoverflow.com/a/37412692
            x = np.arange(len(item['meta'].index))
            xU, xL = x + 0.5, x - 0.5 
            prob = ss.norm.cdf(xU, scale = 3) - ss.norm.cdf(xL, scale = 3)
            prob = prob / prob.sum() # normalize the probabilities so their sum is 1
            index = np.random.choice(x, size=1, p = prob)
            subrecord = item['meta'].iloc[index].squeeze()
        elif self.mode == 'cont':
            # Select index
            val = np.random.rand() * (len(item['meta'].index) - 2)
            index = math.floor(val)
            alpha = val - index

            # Get index's row and the next one
            # as dataframes
            subrecord = deepcopy(item['meta'].iloc[[index]])
            subrecord_next = item['meta'].iloc[[index + 1]]

            # Interpolate
            cols_to_interpolate = LABEL_COLS_ORDERED + ['eeg_label_offset_seconds', 'spectrogram_label_offset_seconds']
            subrecord.loc[:, cols_to_interpolate] = (
                (1 - alpha) * subrecord[cols_to_interpolate].values[0] + 
                alpha * subrecord_next[cols_to_interpolate].values[0]
            )
            
            # Squeeze to Series
            subrecord = subrecord.squeeze()
        else:
            raise ValueError(f'unknown mode {self.mode}')
        
        # Select
        item = self._select_by_subrecord(subrecord, **item)

        return item


class CenterSubrecord(Subrecord):
    def __call__(self, *args, force_apply: bool = False, **item):
        # Get center sub-record
        subrecord = item['meta'].iloc[len(item['meta'].index) // 2]

        # Add fields missing in test
        if 'eeg_label_offset_seconds' not in subrecord:
            subrecord['eeg_label_offset_seconds'] = 0.0
        if 'spectrogram_label_offset_seconds' not in subrecord:
            subrecord['spectrogram_label_offset_seconds'] = 0.0
        
        # Select
        item = self._select_by_subrecord(subrecord, **item)

        return item


class ReshapeToPatches:
    def __call__(self, *args, force_apply: bool = False, **item):
        # s: (T=300, F=400) -> (K=4, T=300, F=100)
        spectrogram = item['spectrogram']
        T, F = spectrogram.shape
        assert F % N_SPECTROGRAM_TILES == 0
        spectrogram = spectrogram.reshape(
            T, 
            N_SPECTROGRAM_TILES, 
            F // N_SPECTROGRAM_TILES
        ).transpose((1, 0, 2))

        eeg = item['eeg']
        if eeg.ndim == 2:
            # e: (T=10000, F=20) -> (T=50, K=200, F=20)
            T, F = eeg.shape
            assert T % MEL_N_FFT == 0
            T_new = T // MEL_N_FFT
            eeg = eeg.reshape(T_new, MEL_N_FFT, F)
        else:
            # Already (T, K, F)
            assert eeg.ndim == 3

        item['spectrogram'] = spectrogram
        item['eeg'] = eeg

        return item


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, *args, force_apply: bool = False, **item):
        for transform in self.transforms:
            item = transform(**item)
        return item


class Normalize:
    def __init__(
        self, 
        eps=1e-6,
    ):
        self.eps = eps

    def __call__(self, *args, force_apply: bool = False, **item):
        # e: (T=300, F=400)
        spectrogram = item['spectrogram']
        spectrogram[np.isnan(spectrogram)] = 0
        spectrogram = np.log10(spectrogram + self.eps)
        min_, max_ = np.quantile(spectrogram, 0.01), np.quantile(spectrogram, 0.99)
        spectrogram = np.clip(spectrogram, min_, max_)
        spectrogram = (spectrogram - min_) / (max_ - min_)
        
        item['spectrogram'] = spectrogram
        
        return item


class FillNan:
    def __init__(self, eeg_fill, spectrogram_fill):
        self.eeg_fill = eeg_fill
        self.spectrogram_fill = spectrogram_fill

    def __call__(self, *args, force_apply: bool = False, **item):
        spectrogram = item['spectrogram']
        spectrogram[np.isnan(spectrogram).any(1)] = self.spectrogram_fill

        eeg = item['eeg']
        eeg[np.isnan(eeg).any(1)] = self.eeg_fill

        item['spectrogram'] = spectrogram
        item['eeg'] = eeg
        
        return item


def plot_to_array(y, img_array):
    x = (np.arange(0, y.shape[0]) / y.shape[0] * img_array.shape[1]).astype(int)
    y = (-y * img_array.shape[0]).astype(int) + img_array.shape[0]
    img_array = jplt.vectorized_lines_with_thickness(
        y[:-1], 
        x[:-1], 
        y[1:], 
        x[1:],
        img_array,
        clr=(255, 255, 255, 255), 
        thickness=1
    )


# https://stackoverflow.com/a/25192640
def butter_lowpass(cutoff, fs, order):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)


def butter_lowpass_filter(data, cutoff=20, fs=EED_SAMPLING_RATE_HZ, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


class ToImage:
    def __init__(self, eeg_norm: Literal['precalc', 'gain'] = 'precalc'):
        self.eeg_norm = eeg_norm
        self.gain_factors = [2 ** i for i in range(10)]

    def __call__(self, *args, force_apply: bool = False, **item):
        # TODO: add adaptive image size
        height = 32
        img_array = np.zeros((height * len(EEG_DIFF_COL_INDICES), 1600, 4), dtype=np.uint16)

        eeg = item['eeg']
        subdf = eeg[3000:7000].T
        for i, cols in enumerate(EEG_DIFF_COL_INDICES):
            if len(cols) == 1:
                y = subdf[cols[0]]
            else:
                y = subdf[cols[0]] - subdf[cols[1]]

            y[np.isnan(y)] = 0

            # Remove 50 Hz & 60 Hz noise
            b, a = butter(5, (59, 61), btype='bandstop', analog=False, fs=EED_SAMPLING_RATE_HZ)
            y = lfilter(b, a, y, axis=0)

            # Bandpass 1-70 Hz
            b, a = butter(5, (1, 70), btype='bandpass', analog=False, fs=EED_SAMPLING_RATE_HZ)
            y = lfilter(b, a, y, axis=0)

            # Normalize
            if self.eeg_norm == 'precalc':
                # Clip to 2 * max
                gain_factor = 1.0
            else:
                # Clip to 2 * max, but scale before that
                abs_max = np.quantile(np.abs(y), 0.95)
                for gain_factor in self.gain_factors:
                    if abs_max <= gain_factor * 2 * EEG_DIFF_ABS_MAX[i]:
                        break
            min_, max_ = \
                -gain_factor * 2 * EEG_DIFF_ABS_MAX[i], \
                gain_factor * 2 * EEG_DIFF_ABS_MAX[i]
            y = np.clip(y, min_, max_)
            y = (y - min_) / (max_ - min_)
            y[0] = 0
            y[-1] = 1

            end = -16
            if i == 0:
                plot_to_array(y, img_array[height * i:height * i + height + height // 2, :])
            elif i == len(EEG_DIFF_COL_INDICES) - 1:
                plot_to_array(y, img_array[height * i - height // 2:height * i + height, :])
            else:
                plot_to_array(y, img_array[height * i - height // 2:height * i + height + height // 2, :])
            
            # Fill gain factor register
            img_array[height * i:height * i + height, end:] = gain_factor / 9 * 255

        img_array = np.clip(img_array, 0, 255)
        img = np.zeros((640, 320+1600), dtype=np.float32)
        img[:, 320:] = img_array[..., 3] / 255.0

        # 10 minutes spectrogram
        y = item['spectrogram']
        y[np.isnan(y)] = 0
        y = cv2.resize(y, (320, 320))
        img[:320, :320] = y

        # 50 seconds EEG spectrogram
        y = item['eeg_spectrogram']
        y = y.transpose(1, 2, 0).reshape(-1, y.shape[0] * y.shape[2])
        y = cv2.resize(y, (320, 320))
        img[320:, :320] = y
    
        item['image'] = img
        return item

class Unsqueeze:
    def __call__(self, *args, force_apply: bool = False, **item):
        item['image'] = np.expand_dims(item['image'], axis=0)
        return item


class RandomLrFlip:
    def __init__(self, always_apply=False, p=0.5):
        self.p = p
        self.always_apply = always_apply

    def __call__(self, *args, force_apply: bool = False, **item):
        if not force_apply and not self.always_apply and random.random() > self.p:
            return item
        assert 'image' not in item, 'RandomFlip should be applied before ToImage'

        # Swap left and right EEGs: odd and even numbered channels
        item['eeg'] = item['eeg'][:, EEG_LR_FLIP_REORDER_INDICES]

        # Flip spectrogram: swap first and last halves
        spectrogram_len = item['spectrogram'].shape[1]
        item['spectrogram'] = np.concatenate(
            [
                item['spectrogram'][:, spectrogram_len // 2:], 
                item['spectrogram'][:, :spectrogram_len // 2]
            ],
            axis=1
        )

        # Flip EEG spectrogram: swap 0 and 2, 1 and 3 in the last dimension
        item['eeg_spectrogram'] = item['eeg_spectrogram'][:, :, EEG_SPECTROGRAM_LR_FLIP_REORDER_INDICES]

        return item


class MixUpHms(A.MixUp):
    def get_params(self) -> Dict[str, Union[None, float, Dict[str, Any]]]:
        mix_idx = random.randint(0, len(self.reference_data) - 1)
        mix_data = self.reference_data[mix_idx]

        # If mix_data is None or empty after the above checks, return default values
        if mix_data is None:
            return {"mix_data": {}, "mix_coef": 1}

        # If mix_data is not None, calculate mix_coef and apply read_fn
        mix_coef = beta(self.alpha, self.alpha)  # Assuming beta is defined elsewhere
        return {"mix_data": self.read_fn(mix_data), "mix_coef": mix_coef}

    def __call__(self, *args: Any, force_apply: bool = False, **kwargs: Any) -> Any:
        kwargs['global_label'] = kwargs['label']
        result = super().__call__(*args, force_apply=force_apply, **kwargs)
        result['label'] = result['global_label']
        del result['global_label']
        return result
