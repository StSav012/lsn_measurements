# -*- coding: utf-8 -*-
from __future__ import annotations

from os import PathLike
from typing import Iterable

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from qtpy.QtCore import Qt
from qtpy.QtGui import QBrush, QColor, QPen
from qtpy.QtWidgets import QWidget

__all__ = ['Histogram']

from backend.utils import warning


class Histogram(pg.PlotWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._plot_line: pg.PlotDataItem | None = None
        self.plotItem.showGrid(x=True, y=True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self._text: str = ''
        self._unit: str = ''
        self._x_log: bool = False
        self._y_log: bool = False

    def set_label(self, text: str, unit: str) -> None:
        self._text = str(text)
        self._unit = str(unit)
        x_axis: pg.AxisItem = self.getAxis('bottom')
        x_axis.setLabel(text=self._text, units=self._unit)
        y_axis: pg.AxisItem = self.getAxis('left')
        y_axis.setLabel(text=self.tr('Density'), units=f'1/{self._unit}')
        y_axis.enableAutoSIPrefix(False)

    def hist(self, data: NDArray[float] | Iterable[float],
             bins: int | Iterable[float] | str = 'auto',
             symbol: str = 'o', name: str | None = None,
             pen: (None | int | float | str
                   | tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int]
                   | QColor | QPen) = 0,
             symbolPen: (None | int | float | str
                         | tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int]
                         | QColor | QPen) = 0,
             symbolBrush: (None | int | float | str
                           | tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int]
                           | QColor | QBrush) = 0) -> pg.PlotDataItem | None:
        if self._plot_line is not None:
            self.removeItem(self._plot_line)
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.size == 0:
            self._plot_line = None
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
                return self._plot_line
            hist, bin_edges = np.histogram(np.log(data), bins=bins, density=False)
            hist = hist.astype(np.float64)
            hist /= (np.exp(bin_edges[1:]) - np.exp(bin_edges[:-1])) * data.size
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            bin_centers = np.exp(bin_centers)
        else:
            hist, bin_edges = np.histogram(data, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        if self._y_log:
            hist = hist.astype(np.float64)
            hist[hist == 0.0] = np.nan
        if np.all(np.isnan(bin_centers)) or np.all(np.isnan(hist)):
            self._plot_line = None
        else:
            self._plot_line = self.plot(bin_centers, hist, name=name, symbol=symbol, pen=pen,
                                        symbolPen=symbolPen, symbolBrush=symbolBrush)
        return self._plot_line

    def clear(self) -> None:
        self.plotItem.clear()
        self._plot_line = None

    def setLogMode(self, x: bool | None = None, y: bool | None = None) -> None:
        if x is not None:
            self._x_log = x
        if y is not None:
            self._y_log = y
        self.plotItem.setLogMode(x=x, y=y)

    def save(self, filename: str | PathLike[str]) -> None:
        if self._plot_line is None:
            return
        dataset: tuple[None, None] | tuple[NDArray[float], NDArray[float]] = self._plot_line.getOriginalDataset()
        if dataset[0] is not None and dataset[1] is not None:
            np_dataset: NDArray[float] = np.column_stack(dataset)
            np_dataset[np.isnan(np_dataset)] = 0.0
            np.savetxt(filename, np_dataset, delimiter='\t')
        else:
            warning('No histogram to save')
