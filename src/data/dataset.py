import pandas as pd
from pathlib import Path
from typing import Callable, Dict


LABEL_COLNAMES = [
    'seizure_vote',
    'lpd_vote',
    'gpd_vote',
    'lrda_vote',
    'grda_vote',
    'other_vote',
]
class HmsDataset:
    def __init__(
        self, 
        df_meta: pd.DataFrame, 
        eeg_dirpath: Path, 
        spectrogram_dirpath: Path, 
        transform: Callable | None = None,
        cache: Dict[Path, pd.DataFrame] | None = None,
    ):
        self.df_meta = df_meta
        self.eeg_dirpath = eeg_dirpath
        self.spectrogram_dirpath = spectrogram_dirpath
        self.transform = transform
        self.index_to_eeg_id = {i: id_ for i, id_ in enumerate(sorted(df_meta['eeg_id'].unique()))}
        if cache is None:
            cache = dict()
        self.cache = cache

    def __len__(self) -> int:
        return len(self.index_to_eeg_id)
    
    def read_parquet(self, path: Path):
        if path not in self.cache:
            self.cache[path] = pd.read_parquet(path)
        return self.cache[path]
    
    def __getitem__(self, idx: int):
        # Get group of same eed_id
        eeg_id = self.index_to_eeg_id[idx]
        df = self.df_meta[self.df_meta['eeg_id'] == eeg_id]
        spectrogram_id = df['spectrogram_id'].iloc[0]  # all spectrogram_id are same 

        # Get item
        item = {
            'eeg': self.read_parquet(self.eeg_dirpath / f'{eeg_id}.parquet'),
            'spectogram': self.read_parquet(self.spectrogram_dirpath / f'{spectrogram_id}.parquet'),
            'label': df[LABEL_COLNAMES],
            'meta': df,
        }

        # Apply transform
        if self.transform is not None:
            item = self.transform(**item)

        return item
