#!/usr/bin/env python3
# https://github.com/gregversteeg/gaussianize/blob/master/gaussianize.py
# -*- coding: utf-8 -*-
"""
Transform data so that it is approximately normally distributed

This code written by Greg Ver Steeg, 2015.
"""

import librosa
import logging
import numpy as np
import os
import pandas as pd
import sklearn
from scipy import special
from scipy.stats import kurtosis, norm, rankdata, boxcox
from scipy import optimize  # TODO: Explore efficacy of other opt. methods
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm
from typing import Text, List, Union
from matplotlib import pylab as plt
from scipy import stats

from .constants import (
    EED_SAMPLING_RATE_HZ, 
    N_EEG_TIME_WINDOW,
    EEG_COLS_ORDERED,
    EEG_FFT_WINDOW_SIZE,
    EEG_GAUSSIANIZE_COEFS_TRAIN,
)


logger = logging.getLogger(__name__)
# Tolerance for == 0.0 tolerance.
_EPS = 1e-6


def _update_x(x: Union[np.ndarray, List]) -> np.ndarray:
    x = np.asarray(x)
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    elif len(x.shape) != 2:
        raise ValueError("Data should be a 1-d list of samples to transform or a 2d array with samples as rows.")
    return x


class Gaussianize(sklearn.base.TransformerMixin):
    """
    Gaussianize data using various methods.

    Conventions
    ----------
    This class is a wrapper that follows sklearn naming/style (e.g. fit(X) to train).
    In this code, x is the input, y is the output. But in the functions outside the class, I follow
    Georg's convention that Y is the input and X is the output (Gaussianized) data.

    Parameters
    ----------
    
    strategy : str, default='lambert'. Possibilities are 'lambert'[1], 'brute'[2] and 'boxcox'[3].

    tol : np.float32, default = 1e-4

    max_iter : int, default = 100
        Maximum number of iterations to search for correct parameters of Lambert transform.

    Attributes
    ----------
    coefs_ : list of tuples
        For each variable, we have transformation parameters.
        For Lambert, e.g., a tuple consisting of (mu, sigma, delta), corresponding to the parameters of the
        appropriate Lambert transform. Eq. 6 and 8 in the paper below.

    References
    ----------
    [1] Georg M Goerg. The Lambert Way to Gaussianize heavy tailed data with
                        the inverse of Tukey's h transformation as a special case
        Author generously provides code in R: https://cran.r-project.org/web/packages/LambertW/
    [2] Valero Laparra, Gustavo Camps-Valls, and Jesus Malo. Iterative Gaussianization: From ICA to Random Rotations
    [3] Box cox transformation and references: https://en.wikipedia.org/wiki/Power_transform
    """

    def __init__(self, strategy: Text = 'lambert', 
                 tol: np.float32 = 1e-5, 
                 max_iter: int = 100, 
                 verbose: bool = False):
        self.tol = tol
        self.max_iter = max_iter
        self.strategy = strategy
        self.coefs_ = []  # Store tau for each transformed variable
        self.verbose = verbose
        
    def fit(self, x: np.ndarray, y=None, names=None):
        """Fit a Gaussianizing transformation to each variable/column in x."""
        # Initialize coefficients again with an empty list.  Otherwise
        # calling .fit() repeatedly will augment previous .coefs_ list.
        self.coefs_ = []
        x = _update_x(x)
        if self.verbose:
            print("Gaussianizing with strategy='%s'" % self.strategy)
        
        if self.strategy == "lambert":
            _get_coef = lambda vec: igmm(vec, self.tol, max_iter=self.max_iter)
        elif self.strategy == "brute":
            _get_coef = lambda vec: None   # TODO: In principle, we could store parameters to do a quasi-invert
        elif self.strategy == "boxcox":
            _get_coef = lambda vec: boxcox(vec)[1]
        else:
            raise NotImplementedError("stategy='%s' not implemented." % self.strategy)

        if names is None:
            names = range(x.shape[1])

        for x_i, name in tqdm(zip(x.T, names)):
            logging.info(f'Started {self.strategy} process for {name}')
            self.coefs_.append(_get_coef(x_i))

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform new data using a previously learned Gaussianization model."""
        x = _update_x(x)
        if x.shape[1] != len(self.coefs_):
            raise ValueError("%d variables in test data, but %d variables were in training data." % (x.shape[1], len(self.coefs_)))

        if self.strategy == 'lambert':
            return np.array([w_t(x_i, tau_i) for x_i, tau_i in zip(x.T, self.coefs_)]).T
        elif self.strategy == 'brute':
            return np.array([norm.ppf((rankdata(x_i) - 0.5) / len(x_i)) for x_i in x.T]).T
        elif self.strategy == 'boxcox':
            return np.array([boxcox(x_i, lmbda=lmbda_i) for x_i, lmbda_i in zip(x.T, self.coefs_)]).T
        else:
            raise NotImplementedError("stategy='%s' not implemented." % self.strategy)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Recover original data from Gaussianized data."""
        if self.strategy == 'lambert':
            return np.array([inverse(y_i, tau_i) for y_i, tau_i in zip(y.T, self.coefs_)]).T
        elif self.strategy == 'boxcox':
            return np.array([(1. + lmbda_i * y_i) ** (1./lmbda_i) for y_i, lmbda_i in zip(y.T, self.coefs_)]).T
        else:
            raise NotImplementedError("Inversion not supported for gaussianization transform '%s'" % self.strategy)

    def qqplot(self, x: np.ndarray, prefix: Text = 'qq', output_dir: Text = "/tmp/"):
        """Show qq plots compared to normal before and after the transform."""
        x = _update_x(x)
        y = self.transform(x)
        n_dim = y.shape[1]
        for i in range(n_dim):
            stats.probplot(x[:, i], dist="norm", plot=plt)
            plt.savefig(os.path.join(output_dir, prefix + '_%d_before.png' % i))
            plt.clf()
            stats.probplot(y[:, i], dist="norm", plot=plt)
            plt.savefig(os.path.join(output_dir, prefix + '_%d_after.png' % i))
            plt.clf()


def w_d(z, delta):
    # Eq. 9
    if delta < _EPS:
        return z
    return np.sign(z) * np.sqrt(np.real(special.lambertw(delta * z ** 2)).astype(np.float32) / delta)


def w_t(y, tau):
    # Eq. 8
    return tau[0] + tau[1] * w_d((y - tau[0]) / tau[1], tau[2])


def inverse(x, tau):
    # Eq. 6
    u = (x - tau[0]) / tau[1]
    return tau[0] + tau[1] * (u * np.exp(u * u * (tau[2] * 0.5)))


def igmm(y: np.ndarray, tol: np.float32 = 1e-6, max_iter: int = 100):
    # Infer mu, sigma, delta using IGMM in Alg.2, Appendix C
    if np.std(y) < _EPS:
        return np.mean(y), np.std(y).clip(_EPS), 0
    delta0 = delta_init(y)
    tau1 = (np.median(y), np.std(y) * (1. - 2. * delta0) ** 0.75, delta0)
    for k in range(max_iter):
        tau0 = tau1
        z = (y - tau1[0]) / tau1[1]
        delta1 = delta_gmm(z)
        x = tau0[0] + tau1[1] * w_d(z, delta1)
        mu1, sigma1 = np.mean(x), np.std(x)
        tau1 = (mu1, sigma1, delta1)

        cur_err = np.linalg.norm(np.array(tau1) - np.array(tau0))
        logger.debug(f'Current error {cur_err}')
        if cur_err < tol:
            logger.info(f'Converged for {k} iterations')
            break
        else:
            if k == max_iter - 1:
                logger.warning(
                    f"Warning: No convergence after {max_iter} iterations "
                    f"with current error {cur_err}. Increase max_iter."
                )
    return tau1


def delta_gmm(z):
    # Alg. 1, Appendix C
    delta0 = delta_init(z)

    def func(q):
        u = w_d(z, np.exp(q))
        if not np.all(np.isfinite(u)):
            return 0.
        else:
            k = kurtosis(u, fisher=True, bias=False)**2
            if not np.isfinite(k) or k > 1e10:
                return 1e10
            else:
                return k

    res = optimize.fmin(func, np.log(delta0), disp=0)
    return np.around(np.exp(res[-1]), 6)


def delta_init(z):
    gamma = kurtosis(z, fisher=False, bias=False)
    with np.errstate(all='ignore'):
        delta0 = np.clip(1. / 66 * (np.sqrt(66 * gamma - 162.) - 6.), 0.01, 0.48)
    if not np.isfinite(delta0):
        delta0 = 0.01
    return delta0


def build_gaussianize(
    df_meta, 
    n_sample=10,
    random_state=123125,
    mode=None
):
    if mode is None:
        gaussianize = None
    elif mode == 'pre':
        # Load coeffs from constants
        gaussianize = Gaussianize()
        gaussianize.coefs_ = np.array(EEG_GAUSSIANIZE_COEFS_TRAIN, dtype=np.float32)
    elif mode == 'online':
        # Sample from EED dataframe
        eeds = []
        for eed_id in tqdm(sorted(df_meta['eeg_id'].unique())):
            eed = pd.read_parquet(f"/workspace/data_external/train_eegs/{eed_id}.parquet")
            eed = eed.dropna(subset=EEG_COLS_ORDERED)
            eeds.append(eed.sample(n_sample, random_state=random_state))
        df_sample = pd.concat(eeds)
        df_sample = df_sample[EEG_COLS_ORDERED]

        # Fit
        gaussianize = Gaussianize(max_iter=100, tol=1e-4)
        gaussianize.fit(df_sample.values, names=df_sample.columns)
    else:
        raise ValueError(f'unknown gaussianize mode {mode}')

    return gaussianize


class Pretransform:
    def __init__(self, do_clip_eeg: bool, gaussianize_eeg: Gaussianize, do_mel_eeg: bool, max_threads: int = 1):
        self.do_clip_eeg = do_clip_eeg
        self.gaussianize_eeg = gaussianize_eeg
        self.do_mel_eeg = do_mel_eeg
        self.max_threads = max_threads
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        is_eeg = 'EKG' in df.columns

        if not is_eeg:
            return df

        # Clip
        if self.do_clip_eeg:
            # Clip to ~ 0.99 quantile
            df[df.columns.difference(['EKG'])] = df[df.columns.difference(['EKG'])].clip(-5000, 5000)
            df['EKG'] = df['EKG'].clip(-10000, 10000)

        # Gaussianize
        if self.gaussianize_eeg is not None:
            df[EEG_COLS_ORDERED] = self.gaussianize_eeg.transform(df[EEG_COLS_ORDERED].values)
    
        # Mel transform
        if self.do_mel_eeg:
            # Convert to array
            eeg = df[EEG_COLS_ORDERED].values

            # https://www.kaggle.com/code/cdeotte/how-to-make-spectrogram-from-eeg
            # (T, F=20) -> (F=20, K=N_EEG_TIME_WINDOW, T'=T//N_EEG_TIME_WINDOW)
            T, F = eeg.shape
            assert T % N_EEG_TIME_WINDOW == 0

            # Fill nan
            nan_mask = np.isnan(eeg).any(1)
            if not nan_mask.all():
                fill = np.nanmean(eeg, axis=0)
                eeg[nan_mask] = fill
            else:
                eeg[:] = 0
            
            # Get spectrogram
            with threadpool_limits(limits=self.max_threads):
                mel_spec = librosa.feature.melspectrogram(
                    y=eeg.T, 
                    sr=EED_SAMPLING_RATE_HZ, 
                    hop_length=N_EEG_TIME_WINDOW, 
                    n_fft=EEG_FFT_WINDOW_SIZE, 
                    n_mels=N_EEG_TIME_WINDOW,
                    fmax=20,
                )

            # Truncate for alignment
            mel_spec = mel_spec[:, :, :-1]

            # Reshape to (T', K, F)
            df = mel_spec.transpose([2, 1, 0])

        return df
