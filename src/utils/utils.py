import joblib
import logging
import numpy as np
import torch
from typing import Dict, Optional, Union
from lightning import Trainer
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Literal, Tuple
from weakref import proxy

from src.data.constants import (
    SPECTROGRAM_COLS_ORDERED, 
    EEG_COLS_ORDERED,
)


logger = logging.getLogger(__name__)

###################################################################
########################## General Utils ##########################
###################################################################

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        """Add argument links to parser.

        Example:
            parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        """
        return



class TrainerWandb(Trainer):
    """Hotfix for wandb logger saving config & artifacts to project root dir
    and not in experiment dir."""
    @property
    def log_dir(self) -> Optional[str]:
        """The directory for the current experiment. Use this to save images to, etc...

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                img = ...
                save_img(img, self.trainer.log_dir)
        """
        if len(self.loggers) > 0:
            if isinstance(self.loggers[0], WandbLogger):
                dirpath = self.loggers[0]._experiment.dir
            elif not isinstance(self.loggers[0], TensorBoardLogger):
                dirpath = self.loggers[0].save_dir
            else:
                dirpath = self.loggers[0].log_dir
        else:
            dirpath = self.default_root_dir

        dirpath = self.strategy.broadcast(dirpath)
        return dirpath


class ModelCheckpointNoSave(ModelCheckpoint):
    def best_epoch(self) -> int:
        # exmple: epoch=10-step=1452.ckpt
        return int(self.best_model_path.split('=')[-2].split('-')[0])
    
    def ith_epoch_score(self, i: int) -> Optional[float]:
        # exmple: epoch=10-step=1452.ckpt
        ith_epoch_filepath_list = [
            filepath 
            for filepath in self.best_k_models.keys()
            if f'epoch={i}-' in filepath
        ]
        
        # Not found
        if not ith_epoch_filepath_list:
            return None
    
        ith_epoch_filepath = ith_epoch_filepath_list[-1]
        return self.best_k_models[ith_epoch_filepath]

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        self._last_global_step_saved = trainer.global_step

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))


class TempSetContextManager:
    def __init__(self, obj, attr, value):
        self.obj = obj
        self.attr = attr
        self.value = value

    def __enter__(self):
        self.old_value = getattr(self.obj, self.attr)
        setattr(self.obj, self.attr, self.value)

    def __exit__(self, *args):
        setattr(self.obj, self.attr, self.old_value)



def state_norm(module: torch.nn.Module, norm_type: Union[float, int, str], group_separator: str = "/") -> Dict[str, float]:
    """Compute each state dict tensor's norm and their overall norm.

    The overall norm is computed over all tensor together, as if they
    were concatenated into a single vector.

    Args:
        module: :class:`torch.nn.Module` to inspect.
        norm_type: The type of the used p-norm, cast to float if necessary.
            Can be ``'inf'`` for infinity norm.
        group_separator: The separator string used by the logger to group
            the tensor norms in their own subfolder instead of the logs one.

    Return:
        norms: The dictionary of p-norms of each parameter's gradient and
            a special entry for the total p-norm of the tensor viewed
            as a single vector.
    """
    norm_type = float(norm_type)
    if norm_type <= 0:
        raise ValueError(f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {norm_type}")

    norms = {
        f"state_{norm_type}_norm{group_separator}{name}": p.data.float().norm(norm_type)
        for name, p in module.state_dict().items()
        if not 'num_batches_tracked' in name
    }
    if norms:
        total_norm = torch.tensor(list(norms.values())).norm(norm_type)
        norms[f"state_{norm_type}_norm_total"] = total_norm
    return norms


def build_stats(
    dataset, 
    filepathes: List[Path], 
    type_: Literal['eeg', 'spectrogram'] = 'eeg'
):
    assert type_ in ['eeg', 'spectrogram']
    cols = SPECTROGRAM_COLS_ORDERED if type_ == 'spectrogram' else EEG_COLS_ORDERED

    # Mean, min, max
    min_, max_ = None, None
    means, counts = [], []
    for filepath in tqdm(filepathes, desc=f'Calculating mean for {type_}'):
        df = dataset.read_parquet(filepath)
        df = df.dropna(subset=cols)
        means.append(df[cols].values.mean(axis=0))
        counts.append(len(df.index))

        if min_ is None:
            min_ = df[cols].values.min(axis=0)
        else:
            min_ = np.minimum(min_, df[cols].values.min(axis=0))
        if max_ is None:
            max_ = df[cols].values.max(axis=0)
        else:
            max_ = np.maximum(max_, df[cols].values.max(axis=0))
    mean = np.average(means, axis=0, weights=counts)

    # Std
    means_sq = []
    for filepath in tqdm(filepathes, desc=f'Calculating std for {type_}'):
        df = dataset.read_parquet(filepath)
        df = df.dropna(subset=cols)
        means_sq.append(((df[cols].values - mean[None, :]) ** 2).mean(axis=0))
    std = np.average(means_sq, axis=0, weights=counts) ** 0.5

    return mean, std, min_, max_


def build_hist(
    dataset, 
    filepathes: List[Path], 
    bins: int,
    range_: List[Tuple[float, float]],
    type_: Literal['eeg', 'spectrogram'] = 'eeg',
):
    assert type_ in ['eeg', 'spectrogram']
    cols = SPECTROGRAM_COLS_ORDERED if type_ == 'spectrogram' else EEG_COLS_ORDERED

    hists, bin_edges = [], []
    for filepath in tqdm(filepathes, desc=f'Calculating hist for {type_}'):
        df = dataset.read_parquet(filepath)
        df = df.dropna(subset=cols)
        if len(hists) == 0:
            for i, col in enumerate(cols):
                h, b = np.histogram(
                    df[col].values,
                    bins=bins,
                    range=range_[i],
                    density=False,
                    weights=None,
                )
                hists.append(h)
                bin_edges.append(b)
        else:
            for i, col in enumerate(cols):
                h, _ = np.histogram(
                    df[col].values,
                    bins=bins,
                    range=range_[i],
                    density=False,
                    weights=None,
                )
                hists[i] += h
    
    return hists, bin_edges


def build_has_outliers(
    dataset, 
    filepathes: List[Path], 
    range_: List[Tuple[float, float]],
    type_: Literal['eeg', 'spectrogram'] = 'eeg'
):
    assert type_ in ['eeg', 'spectrogram']
    cols = SPECTROGRAM_COLS_ORDERED if type_ == 'spectrogram' else EEG_COLS_ORDERED

    has_outlier = []
    for filepath in tqdm(filepathes, desc=f'Calculating hist for {type_}'):
        df = dataset.read_parquet(filepath)
        df = df.dropna(subset=cols)

        values = []
        for i, col in enumerate(cols):
            values.append(
                (
                    (df[col].values < range_[i][0]) | 
                    (df[col].values > range_[i][1])
                ).any()
            )
        has_outlier.append(any(values))
    
    return has_outlier



class CacheDictWithSave(dict):
    """Cache dict that saves itself to disk when full."""
    def __init__(self, indices, cache_save_path: Optional[Path] = None, *args, **kwargs):
        assert len(set(indices)) == len(indices)

        self.indices = indices
        self.cache_save_path = cache_save_path
        self.cache_already_on_disk = False
        
        super().__init__(*args, **kwargs)

        if self.cache_save_path is not None and self.cache_save_path.exists():
            logger.info(f'Loading cache from {self.cache_save_path}')
            self.load()
            assert len(self) >= len(indices), \
                f'Cache loaded from {self.cache_save_path} has {len(self)} records, ' \
                f'but {len(indices)} were expected.'

            assert all(index in self for index in indices)

    def __setitem__(self, index, value):
        # Hack to allow setting items in joblib.load()
        initialized = (
            hasattr(self, 'indices') and
            hasattr(self, 'cache_save_path') and
            hasattr(self, 'cache_already_on_disk')
        )
        if not initialized:
            super().__setitem__(index, value)
            return
        
        if len(self) >= len(self.indices) + 1:
            logger.warning(
                f'More records than expected '
                f'({len(self)} >= {len(self.indices) + 1}) '
                f'in cache. Will be added, but not saved to disk.'
            )
        super().__setitem__(index, value)
        if (
            not self.cache_already_on_disk and 
            len(self) >= len(self.indices) and 
            self.cache_save_path is not None
        ):
            self.save()

    def load(self):
        cache = joblib.load(self.cache_save_path)
        self.update(cache)
        self.cache_already_on_disk = True

    def save(self):
        assert not self.cache_already_on_disk, \
            f'cache_already_on_disk = True, but save() was called. ' \
            f'This should not happen.'
        assert not self.cache_save_path.exists(), \
            f'Cache save path {self.cache_save_path} already exists ' \
            f'but was not loaded from disk (cache_already_on_disk = False). ' \
            f'This should not happen.'

        logger.info(f'Saving cache to {self.cache_save_path}')
        joblib.dump(self, self.cache_save_path)
        self.cache_already_on_disk = True
