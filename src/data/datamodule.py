import hashlib
import git
import logging
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

from src.data.dataset import HmsDataset
from src.data.transforms import (
    RandomSubrecord,
    CenterSubrecord,
    ReshapeToPatches,
    Compose,
)
from src.utils.utils import (
    CacheDictWithSave,
    hms_collate_fn,
)


logger = logging.getLogger(__name__)


class HmsDatamodule(LightningDataModule):
    """Base datamodule for HMS data."""
    def __init__(
        self,
        dataset_dirpath: Path,	
        split_index: int,
        n_splits: int = 5,
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

    def build_transforms(self) -> None:   
        self.train_transform = Compose(
            [
                RandomSubrecord(),
                ReshapeToPatches(),
            ]
        )
        self.val_transform = self.test_transform = Compose(
            [
                CenterSubrecord(),
                ReshapeToPatches(),
            ]
        )

    def make_cache(self, parquet_filepathes) -> None:
        cache_save_path = None
        if self.hparams.cache_dir is not None:
            # Name the cache with md5 hash of 
            # /workspace/contrails/src/data/datasets.py file
            # and ContrailsDataset parameters
            # to avoid using cache when the dataset handling 
            # is changed.
            with open(Path(__file__).parent / 'dataset.py', 'rb') as f:
                datasets_content = f.read()
                datasets_file_hash = hashlib.md5(
                    datasets_content + 
                    str(self.hparams.load_kwargs).encode()
                ).hexdigest()
            cache_save_path = self.hparams.cache_dir / f'{datasets_file_hash}.joblib'

        self.cache = CacheDictWithSave(
            indices=parquet_filepathes,
            cache_save_path=cache_save_path,
        )

        if cache_save_path is None:
            return
        
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

    def setup(self, stage: str = None) -> None:
        # Read metadata & prepare filepathes
        df_meta = pd.read_csv(self.hparams.dataset_dirpath / 'train.csv')
        parquet_filepathes = [
            self.hparams.dataset_dirpath / 'train_spectrograms' / f'{spectrogram_id}.parquet'
            for spectrogram_id in df_meta['spectrogram_id'].unique()
        ] + [
            self.hparams.dataset_dirpath / 'train_eegs' / f'{eeg_id}.parquet'
            for eeg_id in df_meta['eeg_id'].unique()
        ]

        self.build_transforms()
        self.make_cache(
            parquet_filepathes=parquet_filepathes,
        )

        # Split to train, val and test
        kfold = StratifiedGroupKFold(n_splits=self.hparams.n_splits, shuffle=False, random_state=None)
        train_indices, val_indices = list(
            kfold.split(
                X=df_meta, 
                y=df_meta['expert_consensus'], 
                groups=df_meta['patient_id']
            )
        )[self.hparams.split_index]
        df_meta_train, df_meta_val = df_meta.iloc[train_indices], df_meta.iloc[val_indices]

        if self.train_dataset is None:
            self.train_dataset = HmsDataset(
                df_meta_train,
                eeg_dirpath=self.hparams.dataset_dirpath / 'train_eegs',
                spectrogram_dirpath=self.hparams.dataset_dirpath / 'train_spectrograms',
                transform=self.train_transform,
                cache=self.cache,
            )

        if self.val_dataset is None:
            self.val_dataset = HmsDataset(
                df_meta_val,
                eeg_dirpath=self.hparams.dataset_dirpath / 'train_eegs',
                spectrogram_dirpath=self.hparams.dataset_dirpath / 'train_spectrograms',
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
