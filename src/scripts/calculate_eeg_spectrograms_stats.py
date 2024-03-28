import argparse
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.scripts.convert_eeg_to_spectrograms import spectrograms_from_eeg


logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format = '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='For each record in the meta file, calculate EEG spect and stats')
    parser.add_argument('meta_filepath', type=Path, help='Meta file path')
    parser.add_argument('eeg_dirpath', type=Path, help='EEGs dir path')
    parser.add_argument('output_dirpath', type=Path, help='Output dir path')
    parser.add_argument('--bins', type=int, default=1000, help='Number of bins for histogram')
    parser.add_argument('--debug', action='store_true', help='Debug mode, only use a subset of the data')
    return parser.parse_args()


def main(args):
    # Checks
    assert args.meta_filepath.exists(), f'File not found: {args.meta_filepath}'
    assert args.eeg_dirpath.is_dir(), f'Dir not found: {args.eeg_dirpath}'

    # Create output dir
    args.output_dirpath.mkdir(parents=True, exist_ok=True)

    # Read meta file
    df_meta = pd.read_csv(args.meta_filepath)

    # Debug mode
    if args.debug:
        df_meta = df_meta.head(100)

    # First pass: min and max
    min_ = np.inf
    max_ = -np.inf
    for eeg_id in tqdm(sorted(df_meta['eeg_id'].unique())):
        df_eeg = pd.read_parquet(args.eeg_dirpath / f'{eeg_id}.parquet')
        img = spectrograms_from_eeg(df_eeg)
        min_ = min(min_, np.nanmin(img))
        max_ = max(max_, np.nanmax(img))
    logger.info(f'min: {min_}, max: {max_}')

    # Second pass: hist
    hist, bin_edges = None, None
    for eeg_id in tqdm(sorted(df_meta['eeg_id'].unique())):
        df_eeg = pd.read_parquet(args.eeg_dirpath / f'{eeg_id}.parquet')
        img = spectrograms_from_eeg(df_eeg)
        h, b = np.histogram(
            img,
            bins=args.bins,
            range=(min_, max_),
            density=False,
            weights=None,
        )
        if hist is None:
            hist = h
            bin_edges = b
        else:
            hist += h
    logger.info(f'hist: {hist}, bin_edges: {bin_edges}')

    # Save
    logger.info(f'Saving hist and bin_edges')
    np.save(args.output_dirpath / 'hist.npy', hist)
    np.save(args.output_dirpath / 'bin_edges.npy', bin_edges)


if __name__ == '__main__':
    args = parse_args()
    main(args)
