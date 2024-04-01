import numpy as np
import pandas as pd
from copy import deepcopy
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
        eeg_spectrograms: Dict[str, np.ndarray] | None = None, 
        pre_transform: Callable | None = None,
        select_transform: Callable | None = None,
        transform: Callable | None = None,
        do_aux_transform: bool = False,
        cache: Dict[Path, pd.DataFrame] | None = None,
        by_subrecord: bool = False,
    ):
        self.df_meta = df_meta
        self.eeg_dirpath = eeg_dirpath
        self.spectrogram_dirpath = spectrogram_dirpath
        self.eeg_spectrograms = eeg_spectrograms
        self.pre_transform = pre_transform
        self.select_transform = select_transform
        self.transform = transform
        self.index_to_eeg_id = {i: id_ for i, id_ in enumerate(sorted(df_meta['eeg_id'].unique()))}
        self.cache = cache
        self.by_subrecord = by_subrecord
        self.do_aux_transform = do_aux_transform
        if self.cache is not None:
            self.populate_cache()

    def __len__(self) -> int:
        if self.by_subrecord:
            return len(self.df_meta)
        else:
            return len(self.index_to_eeg_id)
    
    def __getitem__(self, idx: int):
        if self.by_subrecord:
            # Get single subrecord
            df = self.df_meta.iloc[[idx]]
            eeg_id = df['eeg_id'].iloc[0]
            n_subrecords = (self.df_meta['eeg_id'] == eeg_id).sum()
        else:
            # Get group of same eed_id
            eeg_id = self.index_to_eeg_id[idx]
            df = self.df_meta[self.df_meta['eeg_id'] == eeg_id]
            n_subrecords = 1  # here we will sample single subrecord later
        spectrogram_id = df['spectrogram_id'].iloc[0]

        # Get item
        df_eeg_or_eeg = self.read_parquet(self.eeg_dirpath / f'{eeg_id}.parquet')
        if isinstance(df_eeg_or_eeg, np.ndarray):
            eeg = df_eeg_or_eeg
        else:
            eeg = df_eeg_or_eeg[EEG_COLS_ORDERED].values
        df_spectrogram = self.read_parquet(self.spectrogram_dirpath / f'{spectrogram_id}.parquet')
        eeg_spectrogram = None
        if self.eeg_spectrograms is not None:
            eeg_spectrogram = self.eeg_spectrograms[eeg_id]

        label = None
        if LABEL_COLS_ORDERED[0] in df.columns:
            label = df[LABEL_COLS_ORDERED].values

        item = {
            'eeg': eeg,
            'eeg_spectrogram': eeg_spectrogram,
            'eeg_raw': df_eeg_or_eeg,
            'spectrogram': df_spectrogram[SPECTROGRAM_COLS_ORDERED].values,
            'spectrogram_time': df_spectrogram['time'].values,
            'label': label,
            'meta': df,
            'n_subrecords': n_subrecords,
        }

        # Apply select transform
        if self.select_transform is not None:
            item = self.select_transform(**item)

        # Apply transform
        if self.transform is not None:
            if not self.do_aux_transform:
                item = self.transform(**item)
            else:
                item1 = self.transform(**deepcopy(item))
                item2 = self.transform(**deepcopy(item))
                item = item1
                item['image_aux'] = item2['image']

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
