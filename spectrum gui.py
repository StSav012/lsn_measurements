# coding: utf-8
import sys
from contextlib import suppress
from multiprocessing import Queue
from queue import Empty
from typing import Final, final

import numpy as np
import pyqtgraph as pg
from nidaqmx.system.physical_channel import PhysicalChannel
from numpy.typing import NDArray
from qtpy import QT5
from qtpy.QtCore import QSettings, Qt, QTimer, Slot
from qtpy.QtGui import QCloseEvent, QIcon
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import welch

from hardware import device_adc
from measurement.noise import NoiseMeasurement
from utils import clear_queue_after_process, silent_alive
from utils.processing import get_scipy_signal_windows_by_name

_MAX_ADC_SAMPLE_RATE: Final[float] = device_adc.ai_max_multi_chan_rate


class GUI(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        self.settings: QSettings = QSettings("SavSoft", "Spectrum", self)

        self.setWindowTitle(self.tr("Spectrum"))
        self.setWindowIcon(QIcon("lsn.svg"))

        self.central_widget: QWidget = QWidget(self)
        self.main_layout: QHBoxLayout = QHBoxLayout(self.central_widget)
        self.controls_layout: QVBoxLayout = QVBoxLayout()
        self.parameters_box: QGroupBox = QGroupBox(self.central_widget)
        self.parameters_layout: QFormLayout = QFormLayout(self.parameters_box)
        self.buttons_layout: QHBoxLayout = QHBoxLayout()

        self.figure: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget(self.central_widget)
        self.canvas_trend: pg.PlotItem = self.figure.ci.addPlot(row=0, col=0)
        self.canvas_spectrum: pg.PlotItem = self.figure.ci.addPlot(row=1, col=0)
        self.line_trend: pg.PlotDataItem = self.canvas_trend.plot(np.empty(0), name="")
        self.line_spectrum: pg.PlotDataItem = self.canvas_spectrum.plot(np.empty(0), name="")

        self.combo_channel: pg.ComboBox = pg.ComboBox(self.central_widget)
        self.spin_sample_rate: pg.SpinBox = pg.SpinBox(self.central_widget)
        self.spin_scale: pg.SpinBox = pg.SpinBox(self.central_widget)
        self.check_power_or_magnitude: QCheckBox = QCheckBox(self.central_widget)
        self.combo_welch_window: pg.ComboBox = pg.ComboBox(self.central_widget)
        self.spin_averaging_time_span: QDoubleSpinBox = QDoubleSpinBox(self.central_widget)
        self.spin_display_time_span: QDoubleSpinBox = QDoubleSpinBox(self.central_widget)

        self.button_start: QPushButton = QPushButton(self.central_widget)
        self.button_stop: QPushButton = QPushButton(self.central_widget)

        self.setup_ui_appearance()
        self.load_settings()
        self.setup_actions()

    def setup_ui_appearance(self) -> None:
        ch: PhysicalChannel
        self.combo_channel.setItems({ch.name: ch for ch in device_adc.ai_physical_chans})

        opts: dict[str, bool | str | int]
        opts = {
            "suffix": self.tr("S/s"),
            "siPrefix": True,
            "decimals": 3,
            "dec": True,
            "compactHeight": False,
            "format": "{scaledValue:.{decimals}f}{suffixGap}{siPrefix}{suffix}",
        }
        self.spin_sample_rate.setOpts(**opts)
        self.spin_sample_rate.setRange(1.0, _MAX_ADC_SAMPLE_RATE)

        opts = {
            "siPrefix": True,
            "decimals": 3,
            "dec": True,
            "compactHeight": False,
            "format": "{scaledValue:.{decimals}f}{suffixGap}{siPrefix}{suffix}",
        }
        self.spin_scale.setOpts(**opts)

        self.check_power_or_magnitude.setText(self.tr("Power units"))

        self.combo_welch_window.setItems({key: value for value, key in get_scipy_signal_windows_by_name()})

        self.spin_averaging_time_span.setMinimum(2.0 / _MAX_ADC_SAMPLE_RATE)
        self.spin_averaging_time_span.setMaximum(np.inf)
        self.spin_averaging_time_span.setSingleStep(1.0 / _MAX_ADC_SAMPLE_RATE)
        self.spin_averaging_time_span.setDecimals(max(0, int(np.ceil(np.log10(_MAX_ADC_SAMPLE_RATE)))))
        self.spin_averaging_time_span.setSuffix(self.tr(" s"))

        self.spin_display_time_span.setMinimum(2.0 / _MAX_ADC_SAMPLE_RATE)
        self.spin_display_time_span.setMaximum(np.inf)
        self.spin_display_time_span.setSingleStep(1.0 / _MAX_ADC_SAMPLE_RATE)
        self.spin_display_time_span.setDecimals(max(0, int(np.ceil(np.log10(_MAX_ADC_SAMPLE_RATE)))))
        self.spin_display_time_span.setSuffix(self.tr(" s"))

        self.figure.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.canvas_trend.setLabels(
            left=(self.tr("Voltage"), self.tr("V")),
            bottom=(self.tr("Time"), self.tr("s")),
        )
        self.canvas_trend.showGrid(x=True, y=False)
        self.canvas_spectrum.setLabels(
            left=(self.tr("Voltage PSD"), self.tr("V / sqrt(Hz)")),
            bottom=(self.tr("Frequency"), self.tr("Hz")),
        )
        self.canvas_spectrum.getAxis("left").autoSIPrefix = False
        self.canvas_spectrum.getAxis("bottom").autoSIPrefix = False
        self.canvas_spectrum.setMenuEnabled(enableMenu=False)
        self.canvas_spectrum.setLogMode(x=True, y=True)
        self.canvas_spectrum.showGrid(x=True, y=True)

        self.main_layout.addWidget(self.figure)
        self.main_layout.addLayout(self.controls_layout)
        self.main_layout.setStretch(0, 1)
        self.controls_layout.addWidget(self.parameters_box)
        self.controls_layout.addLayout(self.buttons_layout)

        self.parameters_layout.addRow(self.tr("Channel:"), self.combo_channel)
        self.parameters_layout.addRow(self.tr("Sample rate:"), self.spin_sample_rate)
        self.parameters_layout.addRow(self.tr("Voltage scale:"), self.spin_scale)
        self.parameters_layout.addWidget(self.check_power_or_magnitude)
        self.parameters_layout.addRow(self.tr("Window:"), self.combo_welch_window)
        self.parameters_layout.addRow(self.tr("Time to average over:"), self.spin_averaging_time_span)
        self.parameters_layout.addRow(self.tr("Time span to show:"), self.spin_display_time_span)

        self.buttons_layout.addWidget(self.button_start)
        self.buttons_layout.addWidget(self.button_stop)

        self.button_start.setText(self.tr("Start"))
        self.button_stop.setText(self.tr("Stop"))
        self.button_stop.setDisabled(True)

        self.setCentralWidget(self.central_widget)

    def setup_actions(self):
        self.button_start.clicked.connect(self.on_button_start_clicked)
        self.button_stop.clicked.connect(self.on_button_stop_clicked)

    def load_settings(self) -> None:
        self.restoreGeometry(self.settings.value("windowGeometry", b""))
        self.restoreState(self.settings.value("windowState", b""))

        self.settings.beginGroup("parameters")
        with suppress(ValueError):
            # `ValueError` might occur when there is no such channel present
            self.combo_channel.setText(self.settings.value("channel", self.combo_channel.currentText(), str))
        self.spin_sample_rate.setValue(self.settings.value("sampleRate", 32678.0, float))
        self.spin_scale.setValue(self.settings.value("voltageScale", 100.0, float))
        self.check_power_or_magnitude.setChecked(self.settings.value("powerUnits", True, bool))
        with suppress(ValueError):
            # `ValueError` might occur when there is no such item present
            self.combo_welch_window.setValue(self.settings.value("welchWindow", "hann", str))
        self.spin_averaging_time_span.setValue(self.settings.value("averagingTimeSpan", 20.0, float))
        self.spin_display_time_span.setValue(self.settings.value("displayTimeSpan", 2.0, float))
        self.settings.endGroup()

    def save_settings(self) -> None:
        self.settings.setValue("windowGeometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())

        self.settings.beginGroup("parameters")
        self.settings.setValue("channel", self.combo_channel.currentText())
        self.settings.setValue("sampleRate", self.spin_sample_rate.value())
        self.settings.setValue("voltageScale", self.spin_scale.value())
        self.settings.setValue("powerUnits", self.check_power_or_magnitude.isChecked())
        self.settings.setValue("welchWindow", self.combo_welch_window.value())
        self.settings.setValue("averagingTimeSpan", self.spin_averaging_time_span.value())
        self.settings.setValue("displayTimeSpan", self.spin_display_time_span.value())
        self.settings.endGroup()

        self.settings.sync()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.save_settings()
        event.accept()

    @Slot()
    def on_button_start_clicked(self) -> None:
        self.button_start.setDisabled(True)
        self.parameters_box.setDisabled(True)
        self.button_stop.setEnabled(True)
        self.line_trend.setData([], [])
        self.line_spectrum.setData([], [])

    @Slot()
    def on_button_stop_clicked(self) -> None:
        self.button_stop.setDisabled(True)
        self.parameters_box.setEnabled(True)
        self.button_start.setEnabled(True)


@final
class App(GUI):
    def __init__(self) -> None:
        super().__init__()

        self.timer: QTimer = QTimer(self)
        self.timer.timeout.connect(self.on_timeout)

        self.results_queue: Queue[tuple[float, NDArray[np.float64]]] = Queue()
        self.measurement: NoiseMeasurement | None = None

        self.v: NDArray[np.float64] = np.empty(0)

        self.check_power_or_magnitude.toggled.connect(self.on_check_power_or_magnitude_toggled)
        self.combo_welch_window.currentTextChanged.connect(self.on_combo_welch_window_current_text_changed)
        self.spin_display_time_span.valueChanged.connect(self.on_spin_display_time_span_value_changed)

    @Slot(bool)
    def on_check_power_or_magnitude_toggled(self, checked: bool) -> None:
        if checked:
            self.canvas_spectrum.setLabels(
                left=(self.tr("Voltage PSD"), self.tr("V² / Hz")),
                bottom=(self.tr("Frequency"), self.tr("Hz")),
            )
        else:
            self.canvas_spectrum.setLabels(
                left=(self.tr("Voltage PSD"), self.tr("V / sqrt(Hz)")),
                bottom=(self.tr("Frequency"), self.tr("Hz")),
            )

        if not self.v.size:
            return

        x_data: NDArray[np.float64] | None = self.line_trend.xData
        if x_data is None:
            return
        sample_rate: float = (x_data.shape[0] - 1) / (x_data[-1] - x_data[0])

        self._draw_spectrum(sample_rate)

    @Slot(str)
    def on_combo_welch_window_current_text_changed(self, text: str) -> None:
        if not text:
            return
        if not self.v.size:
            return

        x_data: NDArray[np.float64] | None = self.line_trend.xData
        if x_data is None:
            return
        sample_rate: float = (x_data.shape[0] - 1) / (x_data[-1] - x_data[0])

        self._draw_spectrum(sample_rate)

    @Slot(float)
    def on_spin_display_time_span_value_changed(self, _value: float) -> None:
        if not self.v.size:
            return

        x_data: NDArray[np.float64] | None = self.line_trend.xData
        if x_data is None:
            return
        sample_rate: float = (x_data.shape[0] - 1) / (x_data[-1] - x_data[0])

        self._draw_spectrum(sample_rate)

    @Slot()
    def on_button_start_clicked(self) -> None:
        self.v = np.empty(0)
        super().on_button_start_clicked()
        self.timer.start(40)
        self.measurement = NoiseMeasurement(
            self.results_queue,
            channel=self.combo_channel.value(),
            sample_rate=self.spin_sample_rate.value(),
        )
        self.measurement.start()

    @Slot()
    def on_button_stop_clicked(self) -> None:
        if silent_alive(self.measurement):
            self.measurement.stop()
        clear_queue_after_process(self.measurement, self.results_queue)
        self.measurement = None
        self.timer.stop()
        super().on_button_stop_clicked()

    @Slot()
    def on_timeout(self) -> None:
        v: NDArray[np.float64]
        sample_rate: float = np.nan
        points_to_display: int
        points_for_spectrum: int
        while self.results_queue.qsize():
            try:
                sample_rate, v = self.results_queue.get()
            except Empty:
                break
            else:
                points_to_display = round(sample_rate * self.spin_display_time_span.value())
                points_for_spectrum = round(sample_rate * self.spin_averaging_time_span.value())
                self.v = np.concatenate((self.v, v[0]))[-max(points_to_display, points_for_spectrum) :]

        if not np.isnan(sample_rate):
            points_to_display = round(sample_rate * self.spin_display_time_span.value())
            self.line_trend.setData(
                np.arange(min(self.v.size, points_to_display)) / sample_rate,
                self.v[-points_to_display:] * self.spin_scale.value(),
            )

            self._draw_spectrum(sample_rate)

    def _draw_spectrum(self, sample_rate: float) -> None:
        points_to_display: int = round(sample_rate * self.spin_display_time_span.value())
        points_for_spectrum: int = round(sample_rate * self.spin_averaging_time_span.value())
        if points_for_spectrum <= 0:
            return
        freq: NDArray[np.float64]
        pn_xx: NDArray[np.float64]
        freq, pn_xx = welch(
            self.v[-points_for_spectrum:],
            fs=sample_rate,
            nperseg=min(points_to_display, points_for_spectrum, self.v.shape[0]),
            window=self.combo_welch_window.value(),
        )
        pn_xx *= self.spin_scale.value() ** 2
        if self.check_power_or_magnitude.isChecked():
            self.line_spectrum.setData(freq, pn_xx)
        else:
            self.line_spectrum.setData(freq, np.sqrt(pn_xx))


if __name__ == "__main__":
    app: QApplication = QApplication(sys.argv)
    if QT5:
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    window: App = App()
    window.show()
    app.exec()
