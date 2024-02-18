import numpy as np
import random
from typing import Dict, List, Callable

from .constants import (
    SPECTROGRAM_N_SAMPLES, 
    EED_N_SAMPLES, 
    EED_SAMPLING_RATE_HZ, 
    LABEL_COLS_ORDERED,
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
    def __call__(self, **item):
        # Sample single sub-record
        subrecord = item['meta'].sample()
        
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
        assert F == 400
        spectrogram = spectrogram.reshape(T, 4, F // 4).transpose((1, 0, 2))

        # e: (T=10000, F=20) -> (T=50, K=200, F=20)
        eeg = item['eeg']
        T, F = eeg.shape
        assert T % 200 == 0
        T_new = T // 200
        eeg = eeg.reshape(T_new, 200, F)

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
