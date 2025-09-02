from collections.abc import Iterable
from os import PathLike
from typing import NamedTuple, TypeVar

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from qtpy.QtCore import Qt
from qtpy.QtGui import QBrush, QColor, QPen
from qtpy.QtWidgets import QWidget

from utils import warning

__all__ = ["Histogram"]


PenType = TypeVar(
    "PenType",
    None,
    int,
    float,
    str,
    tuple[int, int],
    tuple[int, int, int],
    tuple[int, int, int, int],
    QColor,
    QPen,
    dict[
        str,
        None | int | float | str | tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int] | QColor,
    ],
)
BrushType = TypeVar(
    "BrushType",
    None,
    int,
    float,
    str,
    tuple[int, int],
    tuple[int, int, int],
    tuple[int, int, int, int],
    QColor,
    QBrush,
    dict[
        str,
        None | int | float | str | tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int] | QColor,
    ],
)


class LastHistParams(NamedTuple):
    data: NDArray[float] | Iterable[float]
    bins: int | Iterable[float] | str = "auto"
    symbol: str = "o"
    name: str | None = None
    pen: PenType = 0
    symbolPen: PenType = 0
    symbolBrush: BrushType = 0


class Histogram(pg.PlotWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._plot_line: pg.PlotDataItem | None = None
        self._plot_line_p_err: pg.PlotDataItem | None = None
        self._plot_line_n_err: pg.PlotDataItem | None = None

        self.plotItem.showGrid(x=True, y=True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self._text: str = ""
        self._unit: str = ""
        self._x_log: bool = False
        self._y_log: bool = False

        self._hist: NDArray[float] = np.empty(0)
        self._bin_centers: NDArray[float] = np.empty(0)
        self._p_err: NDArray[float] = np.empty(0)
        self._n_err: NDArray[float] = np.empty(0)

        self._last_hist: LastHistParams | None = None

    def set_label(self, text: str, unit: str) -> None:
        self._text = str(text)
        self._unit = str(unit)
        x_axis: pg.AxisItem = self.getAxis("bottom")
        x_axis.setLabel(text=self._text, units=self._unit)
        y_axis: pg.AxisItem = self.getAxis("left")
        y_axis.setLabel(text=self.tr("Density"), units=f"1/{self._unit}")
        y_axis.enableAutoSIPrefix(enable=False)

    def hist(
        self,
        data: NDArray[float] | Iterable[float],
        bins: int | Iterable[float] | str = "auto",
        symbol: str = "o",
        name: str | None = None,
        pen: PenType = 0,
        symbolPen: PenType = 0,
        symbolBrush: BrushType = 0,
    ) -> pg.PlotDataItem | None:
        self._last_hist = LastHistParams(
            data=data,
            bins=bins,
            symbol=symbol,
            name=name,
            pen=pen,
            symbolPen=symbolPen,
            symbolBrush=symbolBrush,
        )

        if self._plot_line is not None:
            self.removeItem(self._plot_line)
        if self._plot_line_p_err is not None:
            self.removeItem(self._plot_line_p_err)
        if self._plot_line_n_err is not None:
            self.removeItem(self._plot_line_n_err)
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.size == 0:
            self._plot_line = None
            self._plot_line_p_err = None
            self._plot_line_n_err = None
            return self._plot_line

        hist: NDArray[float]
        bin_edges: NDArray[float]
        bin_centers: NDArray[float]
        if self._x_log:
            if isinstance(bins, Iterable) and not isinstance(bins, str):
                bins = np.array(bins)
                bins = np.log(bins[bins > 0.0])
            data = data[data > 0.0]
            if data.size == 0:
                self._plot_line = None
                self._plot_line_p_err = None
                self._plot_line_n_err = None
                return self._plot_line
            hist, bin_edges = np.histogram(np.log(data), bins=bins, density=False)
            hist = hist.astype(np.float64)
            hist /= (np.exp(bin_edges[1:]) - np.exp(bin_edges[:-1])) * data.size
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            bin_centers = np.exp(bin_centers)
        else:
            hist, bin_edges = np.histogram(data, bins=bins, density=True)
            hist = hist.astype(np.float64)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        self._hist = hist
        self._bin_centers = bin_centers

        # from https://github.com/veusz/veusz/blob/master/veusz/datasets/histo.py
        ratio: NDArray[float] = 1.0 / (data.size * (bin_edges[1:] - bin_edges[:-1]))
        # “Confidence Limits for Small Numbers of Events in Astrophysical Data,” N. Gehrels, ApJ, 303, 336.
        #  https://adsabs.harvard.edu/full/1986ApJ...303..336G
        p_err: NDArray[float] = np.multiply(1.0 + np.sqrt(np.abs(hist + 0.75)), ratio)
        n_err: NDArray[float] = -np.multiply(np.where(hist > 0.25, np.sqrt(np.abs(hist - 0.25)), np.nan), ratio)
        self._p_err = p_err
        self._n_err = n_err

        if self._y_log:
            hist[hist <= 0.0] = np.nan
            p_err[p_err <= 0.0] = np.nan
            n_err[n_err <= 0.0] = np.nan
        if np.all(np.isnan(bin_centers)) or np.all(np.isnan(hist)):
            self._plot_line = None
            self._plot_line_p_err = None
            self._plot_line_n_err = None
        else:
            self._plot_line = self.plotItem.plot(
                bin_centers,
                hist,
                name=name,
                symbol=symbol,
                pen=pen,
                symbolPen=symbolPen,
                symbolBrush=symbolBrush,
            )
            self._plot_line_p_err = self.plotItem.plot(
                bin_centers,
                hist + p_err,
                name=name,
                symbol=symbol,
                pen=pen,
                symbolPen=symbolPen,
                symbolBrush=symbolBrush,
            )
            self._plot_line_n_err = self.plotItem.plot(
                bin_centers,
                hist + n_err,
                name=name,
                symbol=symbol,
                pen=pen,
                symbolPen=symbolPen,
                symbolBrush=symbolBrush,
            )
            # make error lines semi-transparent
            self._plot_line_p_err.setOpacity(0.5)
            self._plot_line_n_err.setOpacity(0.5)
        return self._plot_line

    def clear(self) -> None:
        self.plotItem.clear()
        self._plot_line = None
        self._plot_line_p_err = None
        self._plot_line_n_err = None

    def setLogMode(self, x: bool | None = None, y: bool | None = None) -> None:
        recompute_hist: bool = False

        if x is not None and self._x_log != x:
            recompute_hist |= True
            self._x_log = x
        if y is not None and self._y_log != y:
            recompute_hist |= True
            self._y_log = y
        self.plotItem.setLogMode(x=x, y=y)

        if recompute_hist and self._last_hist is not None:
            self.hist(*self._last_hist)

    def save(self, filename: str | PathLike[str]) -> None:
        if self._hist.size:
            np.savetxt(
                filename,
                np.column_stack((self._bin_centers, self._hist, self._p_err, self._n_err)),
                delimiter="\t",
                header="\t".join(
                    (
                        f"{self._text} [{self._unit}]",
                        f"Density [1/{self._unit}]",
                        f"Positive error [1/{self._unit}]",
                        f"Negative error [1/{self._unit}]",
                    ),
                ),
                comments="",
            )
        else:
            warning("No histogram to save")
