import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from typing import Callable, Dict

from .constants import (
    LABEL_COLS_ORDERED, 
    SPECTROGRAM_COLS_ORDERED, 
    EEG_COLS_ORDERED,
)


class HmsDataset:
    def __init__(
        self, 
        df_meta: pd.DataFrame, 
        eeg_dirpath: Path, 
        spectrogram_dirpath: Path, 
        pre_transform: Callable | None = None,
        transform: Callable | None = None,
        cache: Dict[Path, pd.DataFrame] | None = None,
    ):
        self.df_meta = df_meta
        self.eeg_dirpath = eeg_dirpath
        self.spectrogram_dirpath = spectrogram_dirpath
        self.pre_transform = pre_transform
        self.transform = transform
        self.index_to_eeg_id = {i: id_ for i, id_ in enumerate(sorted(df_meta['eeg_id'].unique()))}
        self.cache = cache
        if self.cache is not None:
            self.populate_cache()

    def __len__(self) -> int:
        return len(self.index_to_eeg_id)
    
    def __getitem__(self, idx: int):
        # Get group of same eed_id
        eeg_id = self.index_to_eeg_id[idx]
        df = self.df_meta[self.df_meta['eeg_id'] == eeg_id]
        spectrogram_id = df['spectrogram_id'].iloc[0]  # all spectrogram_id are same 

        # Get item
        df_eeg_or_eeg = self.read_parquet(self.eeg_dirpath / f'{eeg_id}.parquet')
        if isinstance(df_eeg_or_eeg, np.ndarray):
            eeg = df_eeg_or_eeg
        else:
            eeg = df_eeg_or_eeg[EEG_COLS_ORDERED].values
        df_spectrogram = self.read_parquet(self.spectrogram_dirpath / f'{spectrogram_id}.parquet')
        item = {
            'eeg': eeg,
            'spectrogram': df_spectrogram[SPECTROGRAM_COLS_ORDERED].values,
            'spectrogram_time': df_spectrogram['time'].values,
            'label': df[LABEL_COLS_ORDERED].values,
            'meta': df,
        }

        # Apply transform
        if self.transform is not None:
            item = self.transform(**item)

        return item

    def __load_pre_transform(self, path):
        item = pd.read_parquet(path)
        if self.pre_transform is not None:
            item = self.pre_transform(item)  # not by **kwargs
        return item

    def read_parquet(self, path: Path):
        if self.cache is None:
            item = self.__load_pre_transform(path)
        else:
            if path in self.cache:
                item = self.cache[path]
            else:
                item = self.__load_pre_transform(path)
                self.cache[path] = item
        return item

    def populate_cache(self):
        filepathes=[
            self.eeg_dirpath / f'{eeg_id}.parquet'
            for eeg_id in self.df_meta['eeg_id'].unique()
        ] + [
            self.spectrogram_dirpath / f'{spectrogram_id}.parquet'
            for spectrogram_id in self.df_meta['spectrogram_id'].unique()
        ]
        for filepath in tqdm(filepathes, desc=f'Polulating cache, len is {len(filepathes)}'):
            _ = self.read_parquet(filepath)
