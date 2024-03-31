import hashlib
import git
import logging
import numpy as np
import pandas as pd
import yaml
import albumentations as A
import torch
from copy import copy, deepcopy
from pathlib import Path
from typing import Optional, Dict, Any, Literal, List
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
    RandomLrFlip,
    MixUpHms,
)
from src.data.constants import LABEL_COLS_ORDERED, BAD_EEG_ID_SUB_ID, BAD_EEG_ID
from src.utils.utils import (
    CacheDictWithSave,
    hms_collate_fn,
    hms_zip_collate_fn,
    ZipDataset,
    ZipRandomSampler,
)
from src.data.pretransform import Pretransform, build_gaussianize


logger = logging.getLogger(__name__)


class HmsDatamodule(LightningDataModule):
    """Base datamodule for HMS data."""
    def __init__(
        self,
        dataset_dirpath: Path,	
        split_index: int,
        eeg_spectrograms_filepath: Path | None = None,
        pl_filepathes: Optional[List[Path]] = None,
        n_splits: int = 5,
        random_subrecord_mode: Literal['discrete', 'gauss_discrete', 'cont'] = 'discrete',
        clip_eeg: bool = True,
        label_smoothing_n_voters: int | None = None,
        low_n_voters_strategy: Literal['low', 'high', 'both', 'simultaneous'] = 'high',
        by_subrecord: bool = False,
        by_subrecord_val: bool = False,
        test_is_train: bool = False,
        img_size: Optional[int] = None,
        drop_bad_which: Optional[Literal['all', 'filtered']] = None,
        drop_bad_from: Optional[Literal['train', 'val', 'both']] = None,
        mixup_alpha: Optional[float] = None,
        eeg_norm: Literal['precalc', 'gain'] = 'precalc',
        cache_dir: Optional[Path] = None,
        batch_size: int = 32,
        force_batch_size: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        super().__init__()

        if eeg_spectrograms_filepath is None:
            eeg_spectrograms_filepath = dataset_dirpath / 'eeg_specs.npy'

        self.save_hyperparameters()

        self.train_dataset_low = None
        self.train_dataset_high = None
        self.train_dataset_both = None
        self.train_dataset_mixup_ref = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_select_transform = None
        self.train_transform = None
        self.train_ref_transform = None
        self.train_mixup_transform = None

        self.val_select_transform = None
        self.val_transform = None

        self.test_select_transform = None
        self.test_transform = None

        self.pre_transform = None

        self.cache = None

    def build_transforms(self) -> None:
        resize_transform = []
        if self.hparams.img_size is not None:
            resize_transform = [A.Resize(self.hparams.img_size, self.hparams.img_size)]
        self.train_select_transform = RandomSubrecord(mode=self.hparams.random_subrecord_mode)
        
        # Add mixup transform
        if self.hparams.mixup_alpha is not None:
            self.train_mixup_transform = MixUpHms(
                reference_data=None, 
                read_fn=lambda x: {'image': x['image'], 'global_label': x['label']},
                alpha=self.hparams.mixup_alpha,
                p=0.5,
            )
        else:
            self.train_mixup_transform = A.NoOp()

        self.train_transform = A.Compose(
            [
                Normalize(
                    eps=1e-6
                ),
                RandomLrFlip(p=0.5),
                ToImage(eeg_norm=self.hparams.eeg_norm),
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
                self.train_mixup_transform,
                A.CoarseDropout(
                    max_holes=5, 
                    max_width=64, 
                    max_height=64, 
                    mask_fill_value=0, 
                    p=0.5,
                ),
                *resize_transform,
                Unsqueeze(),
            ]
        )
        self.train_ref_transform = A.Compose(
            [
                Normalize(
                    eps=1e-6
                ),
                RandomLrFlip(p=0.5),
                ToImage(eeg_norm=self.hparams.eeg_norm),
                A.RandomBrightnessContrast(p=0.5, brightness_limit=0.1, contrast_limit=0.1),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=[0.1, 0.3]),
                        A.GaussianBlur(),
                        A.MotionBlur(),
                    ], 
                    p=0.4
                ),
            ]
        )
        self.val_select_transform = self.test_select_transform = CenterSubrecord()
        self.val_transform = self.test_transform = A.Compose(
            [
                Normalize(
                    eps=1e-6
                ),
                ToImage(eeg_norm=self.hparams.eeg_norm),
                *resize_transform,
                Unsqueeze(),
            ]
        )

    def make_cache(self, df_meta) -> None:
        if self.hparams.cache_dir is None:
            return

        # Get parquet filepathes
        parquet_filepathes = [
            self.hparams.dataset_dirpath / 'train_spectrograms' / f'{spectrogram_id}.parquet'
            for spectrogram_id in sorted(df_meta['spectrogram_id'].unique())
        ] + [
            self.hparams.dataset_dirpath / 'train_eegs' / f'{eeg_id}.parquet'
            for eeg_id in sorted(df_meta['eeg_id'].unique())
        ]
        
        # Name the cache with md5 hash of 
        # /workspace/contrails/src/data/datasets.py file
        # and ContrailsDataset parameters
        # to avoid using cache when the dataset handling 
        # is changed.
        with open(Path(__file__).parent / 'dataset.py', 'rb') as f:
            datasets_content = f.read()
        datasets_file_hash = hashlib.md5(
            datasets_content
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
                'commit_id': commit_id,
                'dirty': dirty,
            }
            yaml.dump(cache_info, f, default_flow_style=False)

        # Make a fake dataset to populate the cache
        # and save it to the cache file
        fake_dataset = HmsDataset(
            df_meta,
            eeg_dirpath=self.hparams.dataset_dirpath / 'train_eegs',
            spectrogram_dirpath=self.hparams.dataset_dirpath / 'train_spectrograms',
            eeg_spectrograms=None,
            pre_transform=self.pre_transform,
            transform=None,
            cache=self.cache,
            by_subrecord=self.hparams.by_subrecord,
        )

    def read_meta(self, test=False):
        if test:
            df_meta = pd.read_csv(self.hparams.dataset_dirpath / 'test.csv')
        else:
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
        # Read meta
        df_meta = self.read_meta(test=False)
        df_meta_test = self.read_meta(test=not self.hparams.test_is_train)

        # Load pre-computed EEG spectrograms
        eeg_spectrograms = np.load(self.hparams.eeg_spectrograms_filepath, allow_pickle=True).item()

        # Make cache for all the data
        self.make_cache(df_meta=df_meta)

        # Build transforms
        self.build_transforms()

        # Split only the rows with large number of voters
        low_mask = df_meta['n_voters'] <= 7
        df_meta_train_low = df_meta[low_mask]
        df_meta_high = df_meta[~low_mask]

        # Split to train, val and test
        kfold = StratifiedGroupKFold(n_splits=self.hparams.n_splits, shuffle=False, random_state=None)
        train_indices, val_indices = list(
            kfold.split(
                X=df_meta_high, 
                y=df_meta_high['expert_consensus'], 
                groups=df_meta_high['patient_id']
            )
        )[self.hparams.split_index]
        df_meta_train_high, df_meta_val = df_meta_high.iloc[train_indices].copy(), df_meta_high.iloc[val_indices].copy()

        # Remove patient_id intersection with val
        # from low n_voters part
        df_meta_train_low = df_meta_train_low[
            ~df_meta_train_low['patient_id'].isin(df_meta_val['patient_id'])
        ]

        # Remove bad samples from train
        # BAD_EEG_ID_SUB_ID: list of tuples of bad eeg_id and subrecord_id
        logger.info(
            f'Length before bad samples removal: '
            f'{len(df_meta_train_low)}, '
            f'{len(df_meta_train_high)}, '
            f'{len(df_meta_val)}'
        )
        if self.hparams.drop_bad_which == 'all':
            if self.hparams.drop_bad_from in ['train', 'both']:
                df_meta_train_low = df_meta_train_low[
                    ~df_meta_train_low['eeg_id'].isin(BAD_EEG_ID)
                ]
                df_meta_train_high = df_meta_train_high[
                    ~df_meta_train_high['eeg_id'].isin(BAD_EEG_ID)
                ]
            if self.hparams.drop_bad_from in ['val', 'both']:
                df_meta_val = df_meta_val[
                    ~df_meta_val['eeg_id'].isin(BAD_EEG_ID)
                ]                
        elif self.hparams.drop_bad_which == 'filtered':
            if self.hparams.drop_bad_from in ['train', 'both']:
                df_meta_train_low = df_meta_train_low[
                    ~df_meta_train_low[['eeg_id', 'eeg_sub_id']].apply(
                        lambda x: tuple(x) in BAD_EEG_ID_SUB_ID, axis=1
                    )
                ]
                df_meta_train_high = df_meta_train_high[
                    ~df_meta_train_high[['eeg_id', 'eeg_sub_id']].apply(
                        lambda x: tuple(x) in BAD_EEG_ID_SUB_ID, axis=1
                    )
                ]
            if self.hparams.drop_bad_from in ['val', 'both']:
                df_meta_val = df_meta_val[
                    ~df_meta_val[['eeg_id', 'eeg_sub_id']].apply(
                        lambda x: tuple(x) in BAD_EEG_ID_SUB_ID, axis=1
                    )
                ]
        logger.info(
            f'Length after bad samples removal: '
            f'{len(df_meta_train_low)}, '
            f'{len(df_meta_train_high)}, '
            f'{len(df_meta_val)}'
        )

        # Use pseudolabels for train objects with low number 
        # of voters
        if self.hparams.pl_filepathes is not None:
            logger.info(f'Length before PLs merge: {len(df_meta_train_low)}')

            # Load all PLs
            for i, pl_filepath in enumerate(self.hparams.pl_filepathes):
                df_pl = pd.read_csv(pl_filepath)
                df_meta_train_low = pd.merge(
                    df_meta_train_low,
                    df_pl,
                    on=('eeg_id', 'eeg_sub_id'),
                    suffixes=('', f'_{i}'),
                    how='left',
                )

            for col in LABEL_COLS_ORDERED:
                # Average
                pl_cols = [f'{col}_{i}' for i in range(len(self.hparams.pl_filepathes))]
                df_meta_train_low[col] = np.nanmean(
                    df_meta_train_low[pl_cols].values, 
                    axis=1
                )

                # Drop PL columns
                df_meta_train_low = df_meta_train_low.drop(columns=pl_cols)

            # Drop nans
            df_meta_train_low = df_meta_train_low.dropna(subset=LABEL_COLS_ORDERED)            
            
            logger.info(f'Length after PLs merge: {len(df_meta_train_low)}')

        # Apply label smoothing
        # Note: label smoothing is not applied to pseudolabels
        df_meta_train_high = self.apply_label_smoothing_n_voters(df_meta_train_high)
        df_meta_train_low = self.apply_label_smoothing_n_voters(df_meta_train_low)
        df_meta_train_both = pd.concat([df_meta_train_low, df_meta_train_high], axis=0)

        if self.train_dataset_low is None:
            self.train_dataset_low = HmsDataset(
                df_meta_train_low,
                eeg_dirpath=self.hparams.dataset_dirpath / 'train_eegs',
                spectrogram_dirpath=self.hparams.dataset_dirpath / 'train_spectrograms',
                eeg_spectrograms=eeg_spectrograms,
                pre_transform=self.pre_transform,
                select_transform=self.train_select_transform,
                transform=self.train_transform,
                # simultaneous trainig is used for training with consistency loss
                # which involves applying two different transformation
                # to the same object from low dataset
                do_aux_transform=self.hparams.low_n_voters_strategy == 'simultaneous',
                cache=self.cache,
                by_subrecord=self.hparams.by_subrecord,
            )
        
        if self.train_dataset_high is None:
            self.train_dataset_high = HmsDataset(
                df_meta_train_high,
                eeg_dirpath=self.hparams.dataset_dirpath / 'train_eegs',
                spectrogram_dirpath=self.hparams.dataset_dirpath / 'train_spectrograms',
                eeg_spectrograms=eeg_spectrograms,
                pre_transform=self.pre_transform,
                select_transform=self.train_select_transform,
                transform=self.train_transform,
                do_aux_transform=False,  # only for low
                cache=self.cache,
                by_subrecord=self.hparams.by_subrecord,
            )

        if self.train_dataset_both is None:
            self.train_dataset_both = HmsDataset(
                df_meta_train_both,
                eeg_dirpath=self.hparams.dataset_dirpath / 'train_eegs',
                spectrogram_dirpath=self.hparams.dataset_dirpath / 'train_spectrograms',
                eeg_spectrograms=eeg_spectrograms,
                pre_transform=self.pre_transform,
                select_transform=self.train_select_transform,
                transform=self.train_transform,
                do_aux_transform=False,  # only for low
                cache=self.cache,
                by_subrecord=self.hparams.by_subrecord,
            )

        # Reference dataset for mixup
        if self.hparams.low_n_voters_strategy == 'low':
            self.train_dataset_mixup_ref = copy(self.train_dataset_low)
        elif self.hparams.low_n_voters_strategy in ['high', 'simultaneous']:
            self.train_dataset_mixup_ref = copy(self.train_dataset_high)
        elif self.hparams.low_n_voters_strategy == 'both':
            self.train_dataset_mixup_ref = copy(self.train_dataset_both)
        self.train_dataset_mixup_ref.df_meta = deepcopy(self.train_dataset_mixup_ref.df_meta)
        self.train_dataset_mixup_ref.df_meta = self.train_dataset_mixup_ref.df_meta[
            self.train_dataset_mixup_ref.df_meta['expert_consensus'] != 'Other'
        ]
        self.train_dataset_mixup_ref.index_to_eeg_id = {i: id_ for i, id_ in enumerate(sorted(self.train_dataset_mixup_ref.df_meta['eeg_id'].unique()))}
        self.train_dataset_mixup_ref.transform = self.train_ref_transform
        if self.hparams.mixup_alpha is not None:
            self.train_mixup_transform.reference_data = self.train_dataset_mixup_ref

        if self.val_dataset is None:
            self.val_dataset = HmsDataset(
                df_meta_val,
                eeg_dirpath=self.hparams.dataset_dirpath / 'train_eegs',
                spectrogram_dirpath=self.hparams.dataset_dirpath / 'train_spectrograms',
                eeg_spectrograms=eeg_spectrograms,
                pre_transform=self.pre_transform,
                select_transform=self.val_select_transform,
                transform=self.val_transform,
                do_aux_transform=False,  # only for low
                cache=self.cache,
                by_subrecord=self.hparams.by_subrecord_val,  # val is always with center subrecord
            )
            
        if self.test_dataset is None:
            if self.hparams.test_is_train:
                # Drop all intersections with high train
                logger.info(
                    f'Length before test train intersection removal: '
                    f'{len(df_meta_test)}'
                )
                df_meta_test = df_meta_test[
                    ~df_meta_test['patient_id'].isin(df_meta_train_high['patient_id'])
                ]
                logger.info(
                    f'Length after test train intersection removal: '
                    f'{len(df_meta_test)}'
                )

                eeg_dirpath = self.hparams.dataset_dirpath / 'train_eegs'
                spectrogram_dirpath = self.hparams.dataset_dirpath / 'train_spectrograms'
            else:
                eeg_dirpath = self.hparams.dataset_dirpath / 'test_eegs'
                spectrogram_dirpath = self.hparams.dataset_dirpath / 'test_spectrograms'
            self.test_dataset = HmsDataset(
                df_meta_test,
                eeg_dirpath=eeg_dirpath,
                spectrogram_dirpath=spectrogram_dirpath,
                eeg_spectrograms=eeg_spectrograms,
                pre_transform=self.pre_transform,
                select_transform=self.test_select_transform,
                transform=self.test_transform,
                do_aux_transform=False,  # only for low
                cache=self.cache,
                # test is always by subrecord: 
                # either single subrecord for kaggle or 
                # all subrecords for local analysis
                by_subrecord=True,
            )

    def train_dataloader(self) -> DataLoader:        
        if self.hparams.low_n_voters_strategy != 'simultaneous':
            # Just a single dataset either with no low n_voters or with them
            if self.hparams.low_n_voters_strategy == 'low':
                train_dataset = self.train_dataset_low
            elif self.hparams.low_n_voters_strategy == 'high':
                train_dataset = self.train_dataset_high
            elif self.hparams.low_n_voters_strategy == 'both':
                train_dataset = self.train_dataset_both
            else:
                raise ValueError(f"Unknown low_n_voters_strategy: {self.hparams.low_n_voters_strategy}")
            shuffle = True
            sampler = None
            collate_fn = hms_collate_fn
        else:
            # Both datasets are used for training
            # simultaneously
            train_dataset = ZipDataset(
                [self.train_dataset_low, self.train_dataset_high]
            )
            shuffle = False

            # Sampler:
            sampler = ZipRandomSampler(
                lengths=[
                    len(self.train_dataset_low), 
                    len(self.train_dataset_high)
                ]
            )
            collate_fn = hms_zip_collate_fn
            
        return DataLoader(
            dataset=train_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory, 
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            sampler=sampler,
            shuffle=shuffle,
            drop_last=True,
            collate_fn=collate_fn,
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
            drop_last=False,
            collate_fn=hms_collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
