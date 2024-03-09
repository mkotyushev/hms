import argparse
import os
import librosa
import logging
import numpy as np
import pandas as pd
import pywt
from pathlib import Path
from tqdm import tqdm

from src.data.constants import (
    EED_SAMPLING_RATE_HZ,
    N_EEG_TIME_WINDOW,
    EEG_FFT_WINDOW_SIZE,
)


logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format = '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='For each record in the meta file, extract corresponding 10s EEGs')
    parser.add_argument('meta_filepath', type=Path, help='Meta file path')
    parser.add_argument('eeg_dirpath', type=Path, help='EEGs dir path')
    parser.add_argument('output_filepath', type=Path, help='Output .npy file path')
    parser.add_argument('--debug', action='store_true', help='Debug mode, only use a subset of the data')
    return parser.parse_args()


USE_WAVELET = None #or "db8" or anything below# DENOISE FUNCTION
NAMES = ['LL','LP','RP','RR']
FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet='haar', level=1):    
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    ret = pywt.waverec(coeff, wavelet, mode='per')
    
    return ret


# https://www.kaggle.com/code/cdeotte/
# how-to-make-spectrogram-from-eeg?scriptVersionId=159854820
def spectrograms_from_eeg(eeg):    
    logger.debug(f'eeg.shape: {eeg.shape}')

    # VARIABLE TO HOLD SPECTROGRAM
    hop_length = 40
    n_fft = 1000
    n_mels = 128
    assert (eeg.shape[0] - n_fft) % hop_length == 0, \
        f'Invalid hop_length: {hop_length} and n_fft: {n_fft} ' \
        f'for eeg.shape[0]: {eeg.shape[0]}'
    img = np.zeros(
        (
            n_mels, 
            (eeg.shape[0] - n_fft) // hop_length + 1, 
            4
        ), 
        dtype=np.float32
    )
    logger.debug(f'img.shape: {img.shape}')
    
    for k in range(4):
        COLS = FEATS[k]
        for kk in range(4):
            # COMPUTE PAIR DIFFERENCES
            x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values

            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean()<1: 
                x = np.nan_to_num(x,nan=m)
            else: 
                x[:] = 0

            # DENOISE
            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(
                y=x, 
                sr=EED_SAMPLING_RATE_HZ, 
                hop_length=hop_length, 
                n_fft=n_fft, 
                n_mels=n_mels,
                fmin=0,
                fmax=20,
                # i-th spectrogram starts at eeg[i*hop_length]
                # and spans n_fft samples
                # https://librosa.org/doc/main/generated/
                # librosa.feature.melspectrogram.html
                center=False,
            )
            logger.debug(f'mel_spec.shape: {mel_spec.shape}')

            # LOG TRANSFORM
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)

            # STANDARDIZE TO -1 TO 1
            mel_spec_db = (mel_spec_db+40)/40 
            img[:,:,k] += mel_spec_db
                
        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:,:,k] /= 4.0
    
    # Convert to 8-bit
    img = (img + 1) / 2 * 255
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def main(args):
    # Checks
    assert args.meta_filepath.exists(), f'File not found: {args.meta_filepath}'
    assert args.eeg_dirpath.is_dir(), f'Dir not found: {args.eeg_dirpath}'
    assert not args.output_filepath.exists(), f'File already exists: {args.output_filepath}'
    assert args.output_filepath.suffix == '.npy', f'Invalid file extension: {args.output_filepath.suffix}'

    # Create output dir
    args.output_filepath.parent.mkdir(parents=True, exist_ok=True)

    # Read meta file
    df_meta = pd.read_csv(args.meta_filepath)

    # Debug mode
    if args.debug:
        df_meta = df_meta.head(100)

    # For each record in the meta file, get spectrograms
    data = dict()
    for eeg_id in tqdm(sorted(df_meta['eeg_id'].unique())):
        df_eeg = pd.read_parquet(args.eeg_dirpath / f'{eeg_id}.parquet')
        data[eeg_id] = spectrograms_from_eeg(df_eeg)

    # Save
    logger.info(f'Saving {len(data)} 50s EEGs spectrograms to {args.output_filepath}')
    np.save(args.output_filepath, data)


if __name__ == '__main__':
    args = parse_args()
    main(args)
