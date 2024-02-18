import math
import numpy as np
import random
from copy import deepcopy
from typing import Dict, List, Callable

from .constants import (
    SPECTROGRAM_N_SAMPLES, 
    EED_N_SAMPLES, 
    EED_SAMPLING_RATE_HZ, 
    LABEL_COLS_ORDERED,
    N_SPECTROGRAM_TILES,
    N_EEG_TIME_WINDOW,
    EEG_COLS_ORDERED,
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
        eeg_start_index = int(subrecord['eeg_label_offset_seconds'] * EED_SAMPLING_RATE_HZ)
        eeg_stop_index = eeg_start_index + EED_N_SAMPLES
        eeg = eeg[eeg_start_index:eeg_stop_index]

        spectrogram = item['spectrogram']
        spectrogram_time = item['spectrogram_time']
        spectrogram_label_offset_seconds = subrecord['spectrogram_label_offset_seconds']
        spectrogram = spectrogram[spectrogram_time >= spectrogram_label_offset_seconds][:SPECTROGRAM_N_SAMPLES]
        spectrogram_time = spectrogram_time[spectrogram_time >= spectrogram_label_offset_seconds][:SPECTROGRAM_N_SAMPLES]

        # Put back to item
        item['eeg'] = eeg
        item['spectrogram'] = spectrogram
        item['spectrogram_time'] = spectrogram_time
        item['label'] = subrecord[LABEL_COLS_ORDERED].values
        item['meta'] = subrecord

        return item


class RandomSubrecord(Subrecord):
    def __init__(self, mode='discrete'):
        self.mode = mode
    
    def __call__(self, **item):
        if self.mode == 'discrete':
            # Sample single sub-record
            subrecord = item['meta'].sample().squeeze()
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
    def __call__(self, **item):
        # Get center sub-record
        subrecord = item['meta'].iloc[len(item['meta'].index) // 2]
        
        # Select
        item = self._select_by_subrecord(subrecord, **item)

        return item


class ReshapeToPatches:
    def __call__(self, **item):
        # s: (T=300, F=400) -> (K=4, T=300, F=100)
        spectrogram = item['spectrogram']
        T, F = spectrogram.shape
        assert F % N_SPECTROGRAM_TILES == 0
        spectrogram = spectrogram.reshape(
            T, 
            N_SPECTROGRAM_TILES, 
            F // N_SPECTROGRAM_TILES
        ).transpose((1, 0, 2))

        # e: (T=10000, F=20) -> (T=50, K=200, F=20)
        eeg = item['eeg']
        T, F = eeg.shape
        assert T % N_EEG_TIME_WINDOW == 0
        T_new = T // N_EEG_TIME_WINDOW
        eeg = eeg.reshape(T_new, N_EEG_TIME_WINDOW, F)

        item['spectrogram'] = spectrogram
        item['eeg'] = eeg

        return item


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, **item):
        for transform in self.transforms:
            item = transform(**item)
        return item


class Normalize:
    def __init__(
        self, 
        eeg_mean, 
        eeg_std,
        spectrogram_mean, 
        spectrogram_std,
        eeg_strategy='meanstd',
        spectrogram_strategy='log',
    ):
        self.eeg_mean = eeg_mean
        self.eeg_std = eeg_std
        self.spectrogram_mean = spectrogram_mean
        self.spectrogram_std = spectrogram_std
        self.eeg_strategy = eeg_strategy
        self.spectrogram_strategy = spectrogram_strategy

    def __call__(self, **item):
        # s: (T=10000, F=20)
        spectrogram = item['spectrogram']
        if self.spectrogram_strategy == 'meanstd':
            spectrogram = (spectrogram - self.spectrogram_mean) / self.spectrogram_std
        elif self.spectrogram_strategy == 'log':
            spectrogram = np.log10(spectrogram + 1)
        else:
            assert self.spectrogram_strategy is None, \
                f'unknown strategy for spectrogram {self.spectrogram_strategy}'

        # e: (T=300, F=400)
        eeg = item['eeg']
        if self.eeg_strategy == 'meanstd':
            eeg = (eeg - self.eeg_mean) / self.eeg_std
        elif self.spectrogram_strategy == 'log':
            abs_, sign = np.abs(eeg), np.sign(eeg)
            eeg = np.log10(abs_ + 1) * sign
        else:
            assert self.eeg_strategy is None, \
                f'unknown strategy for EEG {self.eeg_strategy}'
        
        item['spectrogram'] = spectrogram
        item['eeg'] = eeg
        
        return item


class GaussianizeEegPretransform:
    def __init__(self, gaussianize):
        self.gaussianize = gaussianize
    
    def __call__(self, df):
        if self.gaussianize is None or 'EKG' not in df.columns:
            return df
        df[EEG_COLS_ORDERED] = self.gaussianize.transform(df[EEG_COLS_ORDERED].values)
        return df


class FillNan:
    def __init__(self, eeg_fill, spectrogram_fill):
        self.eeg_fill = eeg_fill
        self.spectrogram_fill = spectrogram_fill

    def __call__(self, **item):
        spectrogram = item['spectrogram']
        spectrogram[np.isnan(spectrogram).any(1)] = self.spectrogram_fill

        eeg = item['eeg']
        eeg[np.isnan(eeg).any(1)] = self.eeg_fill

        item['spectrogram'] = spectrogram
        item['eeg'] = eeg
        
        return item
