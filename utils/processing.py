from collections.abc import Callable, Iterator, Sized
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy import signal

__all__ = ["get_scipy_signal_windows_by_name", "moving_mean", "moving_median", "welch"]

_T = TypeVar("_T")


def welch(
    data: NDArray[np.float64],
    sample_rate: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    freq: NDArray[np.float64]
    pn_xx: NDArray[np.float64]
    freq, pn_xx = signal.welch(data, fs=sample_rate, nperseg=data.size)
    return freq, np.sqrt(pn_xx)


def moving_average(
    x: NDArray[_T],
    n: int,
    averaging_function: Callable[[NDArray[_T]], _T],
) -> NDArray[_T]:
    if n < 0:
        raise ValueError("Window ust have positive length")
    return np.fromiter(
        (averaging_function(x[max(m - n, 0) : m + n]) for m in range(x.size)),
        dtype=x.dtype,
    )


def moving_mean[T](x: NDArray[T], n: int) -> NDArray[T]:
    return moving_average(x, n, np.mean)


def moving_median[T](x: NDArray[T], n: int) -> NDArray[T]:
    return moving_average(x, n, np.median)


def get_scipy_signal_windows_by_name() -> Iterator[tuple[str, str]]:
    from inspect import FullArgSpec, getdoc, getfullargspec

    from scipy.signal import windows

    def none_len(o: Sized | None) -> int:
        if o is None:
            return 0
        return len(o)

    for wn in windows.__dict__.get("__all__", []):
        w: object = getattr(windows, wn)
        if not callable(w):
            continue

        arg_spec: FullArgSpec = getfullargspec(w)
        if none_len(arg_spec.args) + none_len(arg_spec.varargs) - none_len(arg_spec.defaults) != 1 and none_len(
            arg_spec.kwonlyargs,
        ) == none_len(arg_spec.kwonlydefaults):
            continue

        doc: str = getdoc(w)
        if not doc:
            continue

        summary: str = doc.splitlines()[0]
        if summary.startswith("Return") and "window" in summary:
            summary = summary[: summary.index("window") + len("window")]
            summary = " ".join(summary.split()[1:])
            if summary == "a window":
                # fixup, back to the beginning
                summary = " ".join(doc.splitlines()[0].split()[1:]).rstrip(".")
            yield wn, summary
