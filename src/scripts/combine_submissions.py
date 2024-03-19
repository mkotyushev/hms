import argparse
import logging
import pandas as pd
from pathlib import Path

from src.data.constants import LABEL_COLS_ORDERED


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('submission_filepathes', nargs='+', type=Path, help='Submissions to combine')
    parser.add_argument('--output', '-o', type=Path, help='Output file')
    return parser.parse_args()


def main(args):
    logger.info(f'Combining submissions: {args.submission_filepathes}')
    dfs = [pd.read_csv(filepath) for filepath in args.submission_filepathes]
    assert all([df.columns.tolist() == dfs[0].columns.tolist() for df in dfs])
    assert all([(df['eeg_id'] == dfs[0]['eeg_id']).all() for df in dfs])
    
    for filepath, df in zip(args.submission_filepathes, dfs):
        logger.info(
            f'Loaded submission from {filepath} with '
            f'shape: {df.shape}, '
            f'columns: {df.columns.tolist()}'
        )
        logger.info(f'First rows:\n{df.head()}')

    # Mean
    df = dfs[0].copy()
    df[LABEL_COLS_ORDERED] = sum([d[LABEL_COLS_ORDERED].values for d in dfs]) / len(dfs)

    logger.info(f'Combined submission with shape: {df.shape}, columns: {df.columns.tolist()}')
    logger.info(f'First rows:\n{df.head()}')

    # Write
    if args.output:
        df.to_csv(args.output, index=False)
    else:
        print(df)


if __name__ == '__main__':
    args = parse_args()
    main(args)
