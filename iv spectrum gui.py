import sys
from multiprocessing import Queue
from typing import Final, final

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from qtpy import QT5
from qtpy.QtCore import QSettings, QTimer, Qt, Slot
from qtpy.QtGui import QCloseEvent, QIcon
from qtpy.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hardware import device_adc
from measurement.noise import IVNoiseMeasurement
from utils.processing import welch

_MAX_ADC_SAMPLE_RATE: Final[float] = device_adc.ai_max_multi_chan_rate


class GUI(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        self.settings: QSettings = QSettings("SavSoft", "IV Curve", self)

        self.setWindowTitle(self.tr("Spectrum"))
        self.setWindowIcon(QIcon("lsn.svg"))

        self.central_widget: QWidget = QWidget(self)
        self.main_layout: QHBoxLayout = QHBoxLayout(self.central_widget)
        self.controls_layout: QVBoxLayout = QVBoxLayout()
        self.parameters_box: QGroupBox = QGroupBox(self.central_widget)
        self.parameters_layout: QFormLayout = QFormLayout(self.parameters_box)
        self.buttons_layout: QHBoxLayout = QHBoxLayout()

        self.figure: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget(self.central_widget)
        self.canvas_current_trend: pg.PlotItem = self.figure.ci.addPlot(row=0, col=0)
        self.canvas_voltage_trend: pg.PlotItem = self.figure.ci.addPlot(row=1, col=0)
        self.canvas_current_spectrum: pg.PlotItem = self.figure.ci.addPlot(row=0, col=1)
        self.canvas_voltage_spectrum: pg.PlotItem = self.figure.ci.addPlot(row=1, col=1)
        self.current_trend_plot_line: pg.PlotDataItem = self.canvas_current_trend.plot(np.empty(0), name="")
        self.voltage_trend_plot_line: pg.PlotDataItem = self.canvas_voltage_trend.plot(np.empty(0), name="")
        self.current_spectrum_plot_line: pg.PlotDataItem = self.canvas_current_spectrum.plot(np.empty(0), name="")
        self.voltage_spectrum_plot_line: pg.PlotDataItem = self.canvas_voltage_spectrum.plot(np.empty(0), name="")

        self.spin_sample_rate: pg.SpinBox = pg.SpinBox(self.central_widget)
        self.spin_current: pg.SpinBox = pg.SpinBox(self.central_widget)
        self.combo_current_divider: pg.SpinBox = pg.SpinBox(self.central_widget)
        self.spin_ballast_resistance: pg.SpinBox = pg.SpinBox(self.central_widget)
        self.spin_resistance_in_series: pg.SpinBox = pg.SpinBox(self.central_widget)
        self.combo_voltage_gain: pg.ComboBox = pg.ComboBox(self.central_widget)
        self.spin_averaging_time_span: QDoubleSpinBox = QDoubleSpinBox(self.central_widget)
        self.spin_display_time_span: QDoubleSpinBox = QDoubleSpinBox(self.central_widget)

        self.button_start: QPushButton = QPushButton(self.central_widget)
        self.button_stop: QPushButton = QPushButton(self.central_widget)

        self.setup_ui_appearance()
        self.load_settings()
        self.setup_actions()

    def setup_ui_appearance(self) -> None:
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
            "suffix": self.tr("A"),
            "siPrefix": True,
            "decimals": 3,
            "dec": True,
            "compactHeight": False,
            "format": "{scaledValue:.{decimals}f}{suffixGap}{siPrefix}{suffix}",
        }
        self.spin_current.setOpts(**opts)

        opts = {
            "compactHeight": False,
        }
        self.combo_current_divider.setOpts(**opts)

        opts = {
            "suffix": self.tr("Ω"),
            "siPrefix": True,
            "decimals": 6,
            "dec": True,
            "compactHeight": False,
            "format": "{scaledValue:.{decimals}f}{suffixGap}{siPrefix}{suffix}",
        }
        self.spin_ballast_resistance.setOpts(**opts)
        self.spin_resistance_in_series.setOpts(**opts)

        self.combo_voltage_gain.setEditable(True)
        self.combo_voltage_gain.addItems({"2": 2, "10": 10, "100": 100, "1000": 1000})

        self.spin_averaging_time_span.setMinimum(2.0 / _MAX_ADC_SAMPLE_RATE)
        self.spin_averaging_time_span.setMaximum(np.inf)
        self.spin_averaging_time_span.setSingleStep(1.0 / _MAX_ADC_SAMPLE_RATE)
        self.spin_averaging_time_span.setSuffix(self.tr(" s"))

        self.spin_display_time_span.setMinimum(2.0 / _MAX_ADC_SAMPLE_RATE)
        self.spin_display_time_span.setMaximum(np.inf)
        self.spin_display_time_span.setSingleStep(1.0 / _MAX_ADC_SAMPLE_RATE)
        self.spin_display_time_span.setSuffix(self.tr(" s"))

        self.figure.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.canvas_current_trend.setLabels(
            left=(self.tr("Current"), self.tr("A")),
            bottom=(self.tr("Time"), self.tr("s")),
        )
        self.canvas_voltage_trend.setLabels(
            left=(self.tr("Voltage"), self.tr("V")),
            bottom=(self.tr("Time"), self.tr("s")),
        )
        self.canvas_current_trend.showGrid(x=True, y=False)
        self.canvas_voltage_trend.showGrid(x=True, y=False)
        self.canvas_current_spectrum.setLabels(
            left=(self.tr("Current PSD"), self.tr("A / sqrt(Hz)")),
            bottom=(self.tr("Frequency"), self.tr("Hz")),
        )
        self.canvas_voltage_spectrum.setLabels(
            left=(self.tr("Voltage PSD"), self.tr("V / sqrt(Hz)")),
            bottom=(self.tr("Frequency"), self.tr("Hz")),
        )
        self.canvas_current_spectrum.getAxis("left").autoSIPrefix = False
        self.canvas_voltage_spectrum.getAxis("left").autoSIPrefix = False
        self.canvas_current_spectrum.getAxis("bottom").autoSIPrefix = False
        self.canvas_voltage_spectrum.getAxis("bottom").autoSIPrefix = False
        self.canvas_voltage_trend.vb.setXLink(self.canvas_current_trend)
        self.canvas_voltage_spectrum.vb.setXLink(self.canvas_current_spectrum)
        self.canvas_current_spectrum.setMenuEnabled(enableMenu=False)
        self.canvas_voltage_spectrum.setMenuEnabled(enableMenu=False)
        self.canvas_current_spectrum.setLogMode(x=True, y=True)
        self.canvas_voltage_spectrum.setLogMode(x=True, y=True)
        self.canvas_current_spectrum.showGrid(x=True, y=True)
        self.canvas_voltage_spectrum.showGrid(x=True, y=True)

        self.main_layout.addWidget(self.figure)
        self.main_layout.addLayout(self.controls_layout)
        self.main_layout.setStretch(0, 1)
        self.controls_layout.addWidget(self.parameters_box)
        self.controls_layout.addLayout(self.buttons_layout)

        self.parameters_layout.addRow(self.tr("Sample rate:"), self.spin_sample_rate)
        self.parameters_layout.addRow(self.tr("Current:"), self.spin_current)
        self.parameters_layout.addRow(self.tr("Current divider:"), self.combo_current_divider)
        self.parameters_layout.addRow(self.tr("Ballast resistance:"), self.spin_ballast_resistance)
        self.parameters_layout.addRow(self.tr("Resistance in series:"), self.spin_resistance_in_series)
        self.parameters_layout.addRow(self.tr("Voltage gain:"), self.combo_voltage_gain)
        self.parameters_layout.addRow(self.tr("Time to average over"), self.spin_averaging_time_span)
        self.parameters_layout.addRow(self.tr("Time span to show"), self.spin_display_time_span)

        self.buttons_layout.addWidget(self.button_start)
        self.buttons_layout.addWidget(self.button_stop)

        self.button_start.setText(self.tr("Start"))
        self.button_stop.setText(self.tr("Stop"))
        self.button_stop.setDisabled(True)

        self.setCentralWidget(self.central_widget)

    def setup_actions(self) -> None:
        self.button_start.clicked.connect(self.on_button_start_clicked)
        self.button_stop.clicked.connect(self.on_button_stop_clicked)

    def load_settings(self) -> None:
        self.restoreGeometry(self.settings.value("windowGeometry", b""))
        self.restoreState(self.settings.value("windowState", b""))

        self.settings.beginGroup("parameters")
        self.spin_sample_rate.setValue(self.settings.value("sampleRate", 32678.0, float))
        self.spin_current.setValue(self.settings.value("current", 0.0, float))
        self.combo_current_divider.setValue(self.settings.value("currentDivider", 1.0, float))
        self.spin_ballast_resistance.setValue(self.settings.value("ballastResistance", 2e6, float))
        self.spin_resistance_in_series.setValue(self.settings.value("resistanceInSeries", 0.0, float))
        self.combo_voltage_gain.setValue(self.settings.value("voltageGain", 100.0, float))
        self.spin_averaging_time_span.setValue(self.settings.value("averagingTimeSpan", 20.0, float))
        self.spin_display_time_span.setValue(self.settings.value("displayTimeSpan", 2.0, float))
        self.settings.endGroup()

    def save_settings(self) -> None:
        self.settings.setValue("windowGeometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())

        self.settings.beginGroup("parameters")
        self.settings.setValue("sampleRate", self.spin_sample_rate.value())
        self.settings.setValue("current", self.spin_current.value())
        self.settings.setValue("currentDivider", self.combo_current_divider.value())
        self.settings.setValue("ballastResistance", self.spin_ballast_resistance.value())
        self.settings.setValue("resistanceInSeries", self.spin_resistance_in_series.value())
        self.settings.setValue("voltageGain", self.combo_voltage_gain.value())
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
        self.measurement: IVNoiseMeasurement | None = None

        self.i: NDArray[np.float64] = np.empty(0)
        self.v: NDArray[np.float64] = np.empty(0)

    @Slot()
    def on_button_start_clicked(self) -> None:
        super().on_button_start_clicked()
        self.timer.start(40)
        self.measurement = IVNoiseMeasurement(
            self.results_queue,
            sample_rate=self.spin_sample_rate.value(),
            current=self.spin_current.value(),
            current_divider=self.combo_current_divider.value(),
            ballast_resistance=self.spin_ballast_resistance.value(),
            voltage_gain=self.combo_voltage_gain.value(),
            resistance_in_series=self.spin_resistance_in_series.value(),
        )
        self.measurement.start()

    @Slot()
    def on_button_stop_clicked(self) -> None:
        if self.measurement is not None:
            self.measurement.terminate()
            self.measurement.join()
        self.timer.stop()
        super().on_button_stop_clicked()

    @Slot()
    def on_timeout(self) -> None:
        i: NDArray[np.float64]
        v: NDArray[np.float64]
        sample_rate: float = -1.0
        points_to_display: int
        points_for_spectrum: int
        while not self.results_queue.empty():
            sample_rate, (i, v) = self.results_queue.get()
            points_to_display = round(sample_rate * self.spin_display_time_span.value())
            points_for_spectrum = round(sample_rate * self.spin_averaging_time_span.value())
            # v /= self.spin_voltage_gain.value()
            # i -= v
            # i /= self.spin_resistance.value()
            # v -= self.spin_resistance_in_series.value() * i
            self.i = np.concatenate((self.i, i))[-max(points_to_display, points_for_spectrum) :]
            self.v = np.concatenate((self.v, v))[-max(points_to_display, points_for_spectrum) :]
        if sample_rate > 0.0:
            points_to_display = round(sample_rate * self.spin_display_time_span.value())
            points_for_spectrum = round(sample_rate * self.spin_averaging_time_span.value())
            self.current_trend_plot_line.setData(
                np.arange(min(self.i.size, points_to_display)) / sample_rate,
                self.i[-points_to_display:],
            )
            self.voltage_trend_plot_line.setData(
                np.arange(min(self.v.size, points_to_display)) / sample_rate,
                self.v[-points_to_display:],
            )
            self.current_spectrum_plot_line.setData(*welch(self.i[-points_for_spectrum:], sample_rate=sample_rate))
            self.voltage_spectrum_plot_line.setData(*welch(self.v[-points_for_spectrum:], sample_rate=sample_rate))


if __name__ == "__main__":
    app: QApplication = QApplication(sys.argv)
    if QT5:
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    window: App = App()
    window.show()
    app.exec()
