# -*- coding: utf-8 -*-
from typing import Callable, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy import signal

__all__ = ['welch', 'moving_mean', 'moving_median']

_T = TypeVar('_T')


def welch(data: NDArray[np.float64], sample_rate: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    freq: NDArray[np.float64]
    pn_xx: NDArray[np.float64]
    freq, pn_xx = signal.welch(data, fs=sample_rate, nperseg=data.size)
    return freq, np.sqrt(pn_xx)


def moving_average(x: NDArray[_T], n: int, averaging_function: Callable[[NDArray[_T]], _T]) -> NDArray[_T]:
    assert n > 0
    m: int
    return np.fromiter((averaging_function(x[max(m - n, 0):m + n]) for m in range(x.size)), dtype=x.dtype)


def moving_mean(x: NDArray[_T], n: int) -> NDArray[_T]:
    return moving_average(x, n, np.mean)


def moving_median(x: NDArray[_T], n: int) -> NDArray[_T]:
    return moving_average(x, n, np.median)
