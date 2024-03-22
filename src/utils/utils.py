import joblib
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, Optional, Union
from lightning import Trainer
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint, BasePredictionWriter
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import default_collate
from typing import Dict, List, Literal, Tuple
from weakref import proxy

from src.data.constants import (
    SPECTROGRAM_COLS_ORDERED, 
    EEG_COLS_ORDERED,
    LABEL_COLS_ORDERED,
)
from src.data.pretransform import Gaussianize


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

    def before_instantiate_classes(self) -> None:
        # Set LR: nested dict value setting from CLI is not supported
        # so separate arg is used
        if 'fit' in self.config and self.config['fit']['model']['init_args']['lr'] is not None:
            self.config['fit']['model']['init_args']['optimizer_init']['init_args']['lr'] = \
                self.config['fit']['model']['init_args']['lr']


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
            if isinstance(self.loggers[0], WandbLogger) and self.loggers[0]._experiment is not None:
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
    type_: Literal['eeg', 'spectrogram'] = 'eeg',
):
    assert type_ in ['eeg', 'spectrogram']
    cols = SPECTROGRAM_COLS_ORDERED if type_ == 'spectrogram' else EEG_COLS_ORDERED

    # Mean, min, max
    min_, max_ = None, None
    means, counts = [], []
    for filepath in tqdm(filepathes, desc=f'Calculating mean for {type_}'):
        df = dataset.read_parquet(filepath)
        df = df.dropna(subset=cols)
        values = df[cols].values

        means.append(values.mean(axis=0))
        counts.append(len(df.index))

        if min_ is None:
            min_ = values.min(axis=0)
        else:
            min_ = np.minimum(min_, values.min(axis=0))
        if max_ is None:
            max_ = values.max(axis=0)
        else:
            max_ = np.maximum(max_, values.max(axis=0))
    mean = np.average(means, axis=0, weights=counts)

    # Std
    means_sq = []
    for filepath in tqdm(filepathes, desc=f'Calculating std for {type_}'):
        df = dataset.read_parquet(filepath)
        df = df.dropna(subset=cols)
        values = df[cols].values

        means_sq.append(((values - mean[None, :]) ** 2).mean(axis=0))
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


def hms_collate_fn(batch):
    """Collate function for hoa dataset.
    batch: list of dicts of key:str, value: np.ndarray | list | None
    output: dict of torch.Tensor
    """
    output = defaultdict(list)
    for sample in batch:
        for k, v in sample.items():
            if v is None:
                continue

            output[k].append(v)
    
    for k, v in output.items():
        if isinstance(v[0], pd.Series):
            output[k] = pd.DataFrame(v)
        elif isinstance(v[0], np.ndarray):
            v = np.stack(v).astype(np.float32)
            v = default_collate(v)
            output[k] = v
        elif isinstance(v, list):
            output[k] = default_collate(v)
        else:
            output[k] = default_collate(v.astype(np.float32))
    
    return output


def patch_first_conv(model, new_in_channels, default_in_channels=3, pretrained=True, conv_type=nn.Conv2d):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    if new_in_channels == default_in_channels:
        return

    # get first conv
    for module in model.modules():
        if isinstance(module, conv_type) and module.in_channels == default_in_channels:
            break

    weight = module.weight.detach()
    module.in_channels = new_in_channels

    if not pretrained:
        module.weight = nn.parameter.Parameter(
            torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)
        )
        module.reset_parameters()
    elif new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)
    else:
        new_weight = torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)

        for i in range(new_in_channels):
            new_weight[:, i] = weight[:, i % default_in_channels]

        new_weight = new_weight * (default_in_channels / new_in_channels)
        module.weight = nn.parameter.Parameter(new_weight)


class HmsPredictionWriter(BasePredictionWriter):
    def __init__(
        self, 
        output_filepath: Path, 
        image_output_dirpath: Path | None = None,
        drop_eeg_sub_id: bool = True,
    ):
        super().__init__(write_interval='batch_and_epoch')
        self.output_filepath = output_filepath
        self.image_output_dirpath = image_output_dirpath
        if self.image_output_dirpath is not None:
            self.image_output_dirpath.mkdir(parents=True, exist_ok=True)
        self.drop_eeg_sub_id = drop_eeg_sub_id
        self.preds = defaultdict(list)

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        # Increase precision
        prediction = F.softmax(prediction.to(torch.float32), dim=1)
        self.preds['eeg_id'].append(batch['meta']['eeg_id'])
        self.preds['eeg_sub_id'].append(batch['meta']['eeg_sub_id'])
        self.preds['prediction'].append(prediction.detach().cpu().numpy())

        if self.image_output_dirpath is not None:
            images = batch['image'].detach().cpu().numpy()
            for i in range(len(batch['meta']['eeg_id'])):
                eeg_id = batch['meta']['eeg_id'].iloc[i]
                eeg_sub_id = batch['meta']['eeg_sub_id'].iloc[i]
                img = images[i]

                # Save as npy
                filepath = self.image_output_dirpath / f'{eeg_id}_{eeg_sub_id}.npy'
                np.save(filepath, img)

                # Also save as png
                filepath = self.image_output_dirpath / f'{eeg_id}_{eeg_sub_id}.png'
                img = (img * 255).astype(np.uint8)
                img = img[0]
                img = Image.fromarray(img)
                img.save(filepath)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # Make dataframe
        df = pd.DataFrame(
            {
                'eeg_id': np.concatenate(self.preds['eeg_id']),
                'eeg_sub_id': np.concatenate(self.preds['eeg_sub_id']),
            }
        )
        df[LABEL_COLS_ORDERED] = np.concatenate(self.preds['prediction'], axis=0)

        # Convert type
        df['eeg_id'] = df['eeg_id'].astype(int)
        df['eeg_sub_id'] = df['eeg_sub_id'].astype(int)

        # Drop eeg_sub_id
        if self.drop_eeg_sub_id:
            df = df.drop(columns=['eeg_sub_id'])

        # Save
        df.to_csv(self.output_filepath, index=False)
