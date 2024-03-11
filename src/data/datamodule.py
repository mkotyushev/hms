import hashlib
import git
import logging
import numpy as np
import pandas as pd
import yaml
import albumentations as A
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

from src.data.dataset import HmsDataset
from src.data.transforms import (
    RandomSubrecord,
    CenterSubrecord,
    ToImage,
    Normalize,
    Unsqueeze,
)
from src.data.constants import LABEL_COLS_ORDERED
from src.utils.utils import (
    CacheDictWithSave,
    hms_collate_fn,
    build_stats,
)
from src.data.pretransform import Pretransform, build_gaussianize


logger = logging.getLogger(__name__)


class HmsDatamodule(LightningDataModule):
    """Base datamodule for HMS data."""
    def __init__(
        self,
        dataset_dirpath: Path,	
        split_index: int,
        n_splits: int = 5,
        random_subrecord_mode: Literal['discrete', 'cont'] = 'discrete',
        eeg_norm_strategy: Literal['meanstd', 'log', None] = 'meanstd',
        spectrogram_norm_strategy: Literal['meanstd', 'log'] = 'log',
        label_smoothing_n_voters: int | None = None,
        drop_low_n_voters: Literal['train', 'val', 'all'] | None = None,
        cache_dir: Optional[Path] = None,
        load_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        super().__init__()
        if load_kwargs is None:
            load_kwargs = dict()

        self.save_hyperparameters()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_transform = None
        self.val_transform = None
        self.test_transform = None
        self.pre_transform = None

        self.cache = None

    def build_train_stats(self):
        # EEG stats
        eeg_mean, eeg_std = None, None
        if self.hparams.eeg_norm_strategy == 'meanstd':
            eeg_mean, eeg_std, *_ = build_stats(
                self.train_dataset, 
                filepathes=[
                    self.train_dataset.eeg_dirpath / f'{eeg_id}.parquet'
                    for eeg_id in self.train_dataset.df_meta['eeg_id'].unique()
                ],
                type_='eeg',
            )

        # Spectogram stats
        spectrogram_mean, spectrogram_std = None, None
        if self.hparams.spectrogram_norm_strategy == 'meanstd':
            spectrogram_mean, spectrogram_std, *_ = build_stats(
                self.train_dataset, 
                filepathes=[
                    self.train_dataset.spectrogram_dirpath / f'{spectrogram_id}.parquet'
                    for spectrogram_id in self.train_dataset.df_meta['spectrogram_id'].unique()
                ],
                type_='spectrogram',
            )

        return eeg_mean, eeg_std, spectrogram_mean, spectrogram_std

    def build_transforms(self) -> None:
        eeg_mean, eeg_std, spectrogram_mean, spectrogram_std = 0, 1, 0, 1
        if self.hparams.eeg_norm_strategy is not None:
            eeg_mean, eeg_std, spectrogram_mean, spectrogram_std = \
                self.build_train_stats()
        logger.info(
            f'eeg_mean: {eeg_mean}\n'
            f'eeg_std: {eeg_std}\n'
            f'spectrogram_mean: {spectrogram_mean}\n'
            f'spectrogram_std: {spectrogram_std}'
        )

        self.train_transform = A.Compose(
            [
                RandomSubrecord(mode=self.hparams.random_subrecord_mode),
                Normalize(
                    eeg_mean=eeg_mean, 
                    eeg_std=eeg_std,
                    spectrogram_mean=spectrogram_mean, 
                    spectrogram_std=spectrogram_std,
                    eeg_strategy=self.hparams.eeg_norm_strategy,
                    spectrogram_strategy=self.hparams.spectrogram_norm_strategy,
                ),
                ToImage(),
                A.RandomBrightnessContrast(p=0.5, brightness_limit=0.1, contrast_limit=0.1),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=[0.1, 0.3]),
                        A.GaussianBlur(),
                        A.MotionBlur(),
                    ], 
                    p=0.4
                ),
                # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.CoarseDropout(
                    max_holes=5, 
                    max_width=64, 
                    max_height=64, 
                    mask_fill_value=0, 
                    p=0.5,
                ),
                Unsqueeze(),
            ]
        )
        self.val_transform = self.test_transform = A.Compose(
            [
                CenterSubrecord(),
                Normalize(
                    eeg_mean=eeg_mean, 
                    eeg_std=eeg_std,
                    spectrogram_mean=spectrogram_mean, 
                    spectrogram_std=spectrogram_std,
                    eeg_strategy=self.hparams.eeg_norm_strategy,
                    spectrogram_strategy=self.hparams.spectrogram_norm_strategy,
                ),
                ToImage(),
                Unsqueeze(),
            ]
        )

    def make_cache(self, parquet_filepathes) -> None:
        if self.hparams.cache_dir is None:
            return
        
        # Name the cache with md5 hash of 
        # /workspace/contrails/src/data/datasets.py file
        # and ContrailsDataset parameters
        # to avoid using cache when the dataset handling 
        # is changed.
        with open(Path(__file__).parent / 'dataset.py', 'rb') as f:
            datasets_content = f.read()
        with open(Path(__file__).parent / 'pretransform.py', 'rb') as f:
            pretransform_content = f.read()
        datasets_file_hash = hashlib.md5(
            datasets_content + 
            pretransform_content +
            str(self.hparams.load_kwargs).encode()
        ).hexdigest()
        cache_save_path = self.hparams.cache_dir / f'{datasets_file_hash}.joblib'

        self.cache = CacheDictWithSave(
            indices=parquet_filepathes,
            cache_save_path=cache_save_path,
        )        
        self.hparams.cache_dir.mkdir(parents=True, exist_ok=True)

        # Check that only one cache file is in the cache dir
        # and its name is the same as the one we are going to create
        cache_files = list(self.hparams.cache_dir.glob('*.joblib'))
        assert len(cache_files) <= 1, \
            f"More than one cache files found in {cache_save_path} " \
            "which is not advised due to high disk space consumption. " \
            "Please delete all cache files "
        assert len(cache_files) == 0 or cache_files[0] == cache_save_path, \
            f"Cache file {cache_files[0]} is not the same as the one " \
            f"we are going to create {cache_save_path}. " \
            "Please delete all cache files of previous runs."

        # Copy datasets.py to cache dir and
        # save cache info to a file
        # to ease debugging
        (self.hparams.cache_dir / 'dataset.py').write_bytes(datasets_content)
        
        with open(self.hparams.cache_dir / 'cache_info.yaml', 'w') as f:
            commit_id, dirty = None, None
            try:
                commit_id = git.Repo(search_parent_directories=True).head.object.hexsha
                dirty = git.Repo(search_parent_directories=True).is_dirty()
            except git.exc.InvalidGitRepositoryError:
                logger.warning("Not a git repository")
            
            cache_info = {
                'load_kwargs': self.hparams.load_kwargs, 
                'commit_id': commit_id,
                'dirty': dirty,
            }
            yaml.dump(cache_info, f, default_flow_style=False)

    def read_meta(self):
        df_meta = pd.read_csv(self.hparams.dataset_dirpath / 'train.csv')

        # Add n_voters column
        df_meta['n_voters'] = df_meta[LABEL_COLS_ORDERED].sum(axis=1)

        # Normalize the labels
        df_meta[LABEL_COLS_ORDERED] = \
            df_meta[LABEL_COLS_ORDERED].values / \
            df_meta[LABEL_COLS_ORDERED].sum(axis=1).values[:, None]

        return df_meta

    def build_confusion_matrix(self, df):
        expert_consensus_col = df[
            df['n_voters'] > self.hparams.label_smoothing_n_voters
        ][LABEL_COLS_ORDERED].idxmax(axis=1)
        confusion_matrix = []
        for col in LABEL_COLS_ORDERED:
            mask = (
                (df['n_voters'] > self.hparams.label_smoothing_n_voters) & 
                (expert_consensus_col == col)
            )
            s = (
                df[mask][LABEL_COLS_ORDERED].values * 
                df[mask]['n_voters'].values[:, None]
            ).sum(axis=0)
            s = s / s.sum()
            confusion_matrix.append(s)
        confusion_matrix = pd.DataFrame(
            confusion_matrix,
            columns=[f'confused_w_{c}' for c in LABEL_COLS_ORDERED],
            index=LABEL_COLS_ORDERED,
        )
        assert np.allclose(confusion_matrix.sum(axis=1).values, 1)
        return confusion_matrix

    def apply_label_smoothing_n_voters(self, df):
        if self.hparams.label_smoothing_n_voters is None:
            return df

        # Only apply to the labels with the KL divergence > threshold
        # The idea is from https://www.kaggle.com/competitions/
        # hms-harmful-brain-activity-classification/discussion/477461
        confusion_matrix = self.build_confusion_matrix(df)
        
        mask = df['n_voters'] <= self.hparams.label_smoothing_n_voters
        logger.info(
            f'Applying label smoothing to '
            f'{mask.sum() / mask.shape[0]} share of train data'
        )
        df.loc[
            mask, 
            LABEL_COLS_ORDERED
        ] = df.loc[
            mask, 
            LABEL_COLS_ORDERED
        ].values @ confusion_matrix.values

        return df

    def setup(self, stage: str = None) -> None:
        df_meta = self.read_meta()

        # Read metadata & prepare filepathes
        parquet_filepathes = [
            self.hparams.dataset_dirpath / 'train_spectrograms' / f'{spectrogram_id}.parquet'
            for spectrogram_id in sorted(df_meta['spectrogram_id'].unique())
        ] + [
            self.hparams.dataset_dirpath / 'train_eegs' / f'{eeg_id}.parquet'
            for eeg_id in sorted(df_meta['eeg_id'].unique())
        ]

        self.make_cache(
            parquet_filepathes=parquet_filepathes,
        )

        # Drop the rows with small number of voters: all
        if self.hparams.drop_low_n_voters == 'all':
            df_meta = df_meta[df_meta['n_voters'] > 7]

        # Split to train, val and test
        kfold = StratifiedGroupKFold(n_splits=self.hparams.n_splits, shuffle=False, random_state=None)
        train_indices, val_indices = list(
            kfold.split(
                X=df_meta, 
                y=df_meta['expert_consensus'], 
                groups=df_meta['patient_id']
            )
        )[self.hparams.split_index]
        df_meta_train, df_meta_val = df_meta.iloc[train_indices].copy(), df_meta.iloc[val_indices].copy()

        # Apply label smoothing
        df_meta_train = self.apply_label_smoothing_n_voters(df_meta_train)

        # Drop the rows with small number of : only for either train or val
        if self.hparams.drop_low_n_voters == 'train':
            df_meta_train = df_meta_train[df_meta_train['n_voters'] > 7]
        elif self.hparams.drop_low_n_voters == 'val':
            df_meta_val = df_meta_val[df_meta_val['n_voters'] > 7]

        # Build pre-transform
        self.pre_transform = Pretransform(
            do_clip_eeg=self.hparams.load_kwargs.get('do_clip_eeg', False),
            gaussianize_eeg=build_gaussianize(
                df_meta_train, 
                n_sample=10,
                random_state=123125,
                mode=self.hparams.load_kwargs.get('gaussianize_mode', None),
            ),
            do_mel_eeg=self.hparams.load_kwargs.get('do_mel_eeg', False),
            # librosa is kind of bad with unlimited threads + MP 
            # (as when there is no cache and the pretransform 
            # is performed in dataloader), but if cache is enabled, 
            # it is populated (and the pretransform is called) 
            # in the main process. So, if cache is enabled, raise the
            # threading limit.
            max_threads=1 if self.hparams.cache_dir is None else 11
        )

        # Load pre-computed EEG spectrograms
        eeg_spectrograms = np.load(self.hparams.dataset_dirpath / 'eeg_specs.npy', allow_pickle=True).item()

        if self.train_dataset is None:
            self.train_dataset = HmsDataset(
                df_meta_train,
                eeg_dirpath=self.hparams.dataset_dirpath / 'train_eegs',
                spectrogram_dirpath=self.hparams.dataset_dirpath / 'train_spectrograms',
                eeg_spectrograms=eeg_spectrograms,
                pre_transform=self.pre_transform,
                transform=None,  # Here transform depend on the dataset, so will be set later
                cache=self.cache,
            )
            self.build_transforms()
            self.train_dataset.transform = self.train_transform

        if self.val_dataset is None:
            self.val_dataset = HmsDataset(
                df_meta_val,
                eeg_dirpath=self.hparams.dataset_dirpath / 'train_eegs',
                spectrogram_dirpath=self.hparams.dataset_dirpath / 'train_spectrograms',
                eeg_spectrograms=eeg_spectrograms,
                pre_transform=self.pre_transform,
                transform=self.val_transform,
                cache=self.cache,
            )
        
        
        # TODO: add test dataset
        self.test_dataset = None

    def train_dataloader(self) -> DataLoader:        
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory, 
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=hms_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        val_dataloader = DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
            drop_last=True,
            collate_fn=hms_collate_fn,
        )
        
        return val_dataloader

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
            collate_fn=hms_collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
