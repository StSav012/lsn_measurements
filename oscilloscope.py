import enum
import sys
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import suppress
from functools import partial
from multiprocessing import Queue
from pathlib import Path
from queue import Empty
from typing import Any, Final, final

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from qtpy import QT5
from qtpy.QtCore import QSettings, QTimer, Qt, Slot
from qtpy.QtGui import QCloseEvent, QColor, QIcon, QPalette
from qtpy.QtWidgets import (
    QAbstractButton,
    QApplication,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStyle,
    QWidget,
)

from hardware import device_adc
from measurement.noise import NoiseMeasurement
from utils import all_equally_shaped, clear_queue_after_process, silent_alive

_MAX_ADC_SAMPLE_RATE: Final[float] = device_adc.ai_max_multi_chan_rate

type F64Array = NDArray[np.float64]
type I64Array = NDArray[np.int64]


colors: Sequence[QColor] = [
    QColor(Qt.GlobalColor.yellow),
    QColor(Qt.GlobalColor.green),
    QColor(Qt.GlobalColor.blue).lighter(),
    QColor(Qt.GlobalColor.red),
]


def return_none(c: Callable[[], Any]) -> None:
    c()


class GUI(QMainWindow):
    class Mode(enum.Enum):
        Continuous = enum.auto()
        Single = enum.auto()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        self.settings: QSettings = QSettings("SavSoft", "Oscilloscope", self)

        self.setWindowTitle(self.tr("Oscilloscope"))
        self.setWindowIcon(QIcon("lsn.svg"))

        self.figure: pg.PlotWidget = pg.PlotWidget(self)
        self.canvas: pg.PlotItem = self.figure.plotItem
        self.lines: list[pg.PlotDataItem] = [
            self.canvas.plot(np.empty(0), name=ch.name, pen=color)
            for ch, color in zip(device_adc.ai_physical_chans, colors, strict=False)
        ]

        self.menu_bar: QMenuBar = QMenuBar(self)
        self.menu_file: QMenu = self.menu_bar.addMenu(self.tr("&File"))
        self.menu_view: QMenu = self.menu_bar.addMenu(self.tr("&View"))
        self.menu_help: QMenu = self.menu_bar.addMenu(self.tr("&Help"))

        self.channels_box: QDockWidget = QDockWidget(self)
        self.channels_box.setObjectName("channels_box")
        self.channel_buttons: list[QAbstractButton] = []
        for index, (color, ch) in enumerate(zip(colors, device_adc.ai_physical_chans, strict=False)):
            self.channel_buttons.append(button := QPushButton(ch.name))
            palette: QPalette = button.palette()
            palette.setColor(QPalette.ColorRole.Button, color.darker())
            palette.setColor(QPalette.ColorRole.Light, color)
            if color.lightnessF() > 0.5:
                palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.black)
            elif color.lightnessF() < 0.5:
                palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
            else:
                palette.setColor(
                    QPalette.ColorRole.ButtonText,
                    QColor.fromHsvF(1.0 - color.hsvHueF(), color.hsvSaturationF(), color.valueF(), color.alphaF()),
                )
            button.setPalette(palette)
            button.setCheckable(True)
            button.setChecked(True)
            button.toggled.connect(self.lines[index].setVisible)

        self.parameters_box: QDockWidget = QDockWidget(self)
        self.parameters_box.setObjectName("parameters_box")
        self.combo_mode: pg.ComboBox = pg.ComboBox(self.parameters_box)
        self.combo_trigger_channel: pg.ComboBox = pg.ComboBox(self.parameters_box)
        self.spin_trigger_level: pg.SpinBox = pg.SpinBox(self.parameters_box)
        self.combo_trigger_edge: pg.ComboBox = pg.ComboBox(self.parameters_box)
        self.spin_sample_rate: pg.SpinBox = pg.SpinBox(self.parameters_box)
        self.spin_time_span: pg.SpinBox = pg.SpinBox(self.parameters_box)
        self.spin_time_shift: pg.SpinBox = pg.SpinBox(self.parameters_box)
        self.spin_averaging: QSpinBox = QSpinBox(self.parameters_box)

        self.buttons_box: QDockWidget = QDockWidget(self)
        self.buttons_box.setObjectName("buttons_box")
        self.button_start: QPushButton = QPushButton(self.buttons_box)
        self.button_single: QPushButton = QPushButton(self.buttons_box)
        self.button_stop: QPushButton = QPushButton(self.buttons_box)
        self.mode: GUI.Mode = GUI.Mode.Continuous

        self.setup_ui_appearance()
        self.load_settings()
        self.setup_actions()

    def setup_ui_appearance(self) -> None:
        self.combo_mode.setItems({self.tr("Auto"): "auto", self.tr("Normal"): "normal"})
        self.combo_trigger_channel.setItems({ch.name: ch for ch in device_adc.ai_physical_chans})
        self.combo_trigger_edge.setItems(
            {self.tr("Rising"): "rising", self.tr("Falling"): "falling", self.tr("Any"): "any"},
        )

        opts: dict[str, bool | str | int]
        opts = {
            "suffix": self.tr("V"),
            "siPrefix": True,
            "decimals": 3,
            "dec": True,
            "compactHeight": False,
            "format": "{scaledValue:.{decimals}f}{suffixGap}{siPrefix}{suffix}",
        }
        self.spin_trigger_level.setOpts(**opts)
        self.spin_trigger_level.setRange(min(device_adc.ai_voltage_rngs), max(device_adc.ai_voltage_rngs))
        opts = {
            "suffix": self.tr("S/s"),
            "siPrefix": True,
            "decimals": 3,
            "dec": True,
            "compactHeight": False,
            "format": "{scaledValue:.{decimals}f}{suffixGap}{siPrefix}{suffix}",
        }
        self.spin_sample_rate.setOpts(**opts)
        self.spin_sample_rate.setRange(device_adc.ai_min_rate, _MAX_ADC_SAMPLE_RATE)
        opts = {
            "suffix": self.tr("s"),
            "siPrefix": True,
            "dec": True,
            "compactHeight": False,
            "format": "{scaledValue:.{decimals}f}{suffixGap}{siPrefix}{suffix}",
        }
        self.spin_time_span.setOpts(**opts)
        self.spin_time_shift.setOpts(**opts)

        self.spin_time_span.setMinimum(2.0 / _MAX_ADC_SAMPLE_RATE)
        self.spin_time_span.setMaximum(np.inf)
        self.spin_time_span.setSingleStep(1.0 / _MAX_ADC_SAMPLE_RATE)
        self.spin_time_span.setDecimals(max(0, int(np.ceil(np.log10(_MAX_ADC_SAMPLE_RATE)))))

        self.spin_time_shift.setRange(-self.spin_time_span.value() / 2, self.spin_time_span.value() / 2)
        self.spin_time_shift.setSingleStep(1.0 / _MAX_ADC_SAMPLE_RATE)
        self.spin_time_shift.setDecimals(max(0, int(np.ceil(np.log10(_MAX_ADC_SAMPLE_RATE)))))

        self.spin_averaging.setRange(1, 999)

        self.figure.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.canvas.setLabels(
            left=(self.tr("Voltage"), self.tr("V")),
            bottom=(self.tr("Time"), self.tr("s")),
        )
        self.canvas.showGrid(x=True, y=True)

        self.menu_file.addAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
            self.tr("&Save As…"),
            self.save_data,
        )
        self.menu_file.addSeparator()
        self.menu_file.addAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCloseButton),
            self.tr("&Quit"),
            lambda: return_none(self.close),
        )
        self.menu_view.addAction(self.channels_box.toggleViewAction())
        self.menu_view.addAction(self.parameters_box.toggleViewAction())
        self.menu_view.addAction(self.buttons_box.toggleViewAction())
        self.menu_help.addAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarMenuButton),
            self.tr("About &Qt…"),
            partial(QMessageBox.aboutQt, self),
        )
        self.setMenuBar(self.menu_bar)

        channels_layout: QHBoxLayout = QHBoxLayout()
        for button in self.channel_buttons:
            channels_layout.addWidget(button)
        channels_box_widget: QWidget = QWidget(self.channels_box)
        channels_box_widget.setLayout(channels_layout)
        self.channels_box.setWidget(channels_box_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.channels_box)
        self.channels_box.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.channels_box.setWindowTitle(self.tr("Channels"))

        parameters_layout: QFormLayout = QFormLayout()
        parameters_layout.addRow(self.tr("Mode:"), self.combo_mode)
        parameters_layout.addRow(self.tr("Trigger channel:"), self.combo_trigger_channel)
        parameters_layout.addRow(self.tr("Trigger level:"), self.spin_trigger_level)
        parameters_layout.addRow(self.tr("Trigger edge:"), self.combo_trigger_edge)
        parameters_layout.addRow(self.tr("Sample rate:"), self.spin_sample_rate)
        parameters_layout.addRow(self.tr("Time span:"), self.spin_time_span)
        parameters_layout.addRow(self.tr("Time shift:"), self.spin_time_shift)
        parameters_layout.addRow(self.tr("Averaging:"), self.spin_averaging)
        parameters_box_widget: QWidget = QWidget(self.parameters_box)
        parameters_box_widget.setLayout(parameters_layout)
        self.parameters_box.setWidget(parameters_box_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.parameters_box)
        self.parameters_box.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.parameters_box.setWindowTitle(self.tr("Parameters"))

        buttons_layout: QHBoxLayout = QHBoxLayout()
        buttons_layout.addWidget(self.button_start)
        buttons_layout.addWidget(self.button_single)
        buttons_layout.addWidget(self.button_stop)
        buttons_box_widget: QWidget = QWidget(self.buttons_box)
        buttons_box_widget.setLayout(buttons_layout)
        self.buttons_box.setWidget(buttons_box_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.buttons_box)
        self.buttons_box.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.buttons_box.setWindowTitle(self.tr("Controls"))

        self.button_start.setText(self.tr("Start"))
        self.button_single.setText(self.tr("Single"))
        self.button_stop.setText(self.tr("Stop"))
        self.button_stop.setDisabled(True)

        self.setCentralWidget(self.figure)

    def setup_actions(self) -> None:
        self.button_start.clicked.connect(self.on_button_start_clicked)
        self.button_single.clicked.connect(self.on_button_single_clicked)
        self.button_stop.clicked.connect(self.on_button_stop_clicked)

    def load_settings(self) -> None:
        self.restoreGeometry(self.settings.value("windowGeometry", b""))
        self.restoreState(self.settings.value("windowState", b""))

        self.settings.beginGroup("parameters")
        with suppress(ValueError):
            # `ValueError` might occur when there is no such channel present
            self.combo_mode.setValue(self.settings.value("mode", self.combo_mode.value(), str))
        with suppress(ValueError):
            # `ValueError` might occur when there is no such channel present
            self.combo_trigger_channel.setText(
                self.settings.value("triggerChannel", self.combo_trigger_channel.currentText(), str),
            )
        self.spin_trigger_level.setValue(self.settings.value("triggerLevel", 0.0, float))
        with suppress(ValueError):
            # `ValueError` might occur when there is no such channel present
            self.combo_trigger_edge.setValue(self.settings.value("triggerEdge", self.combo_trigger_edge.value(), str))
        self.spin_sample_rate.setValue(self.settings.value("sampleRate", 32678.0, float))
        self.spin_time_span.setValue(self.settings.value("timeSpan", 2.0, float))
        self.spin_time_shift.setRange(-self.spin_time_span.value() / 2, self.spin_time_span.value() / 2)
        self.spin_time_shift.setValue(self.settings.value("timeShift", 0.0, float))
        self.spin_time_span.setSingleStep(1.0 / self.spin_sample_rate.value())
        self.spin_time_span.setMinimum(2.0 / self.spin_sample_rate.value())
        self.spin_time_shift.setSingleStep(1.0 / self.spin_sample_rate.value())
        self.spin_time_span.setDecimals(max(0, int(np.ceil(np.log10(self.spin_sample_rate.value())))))
        self.spin_time_shift.setDecimals(max(0, int(np.ceil(np.log10(self.spin_sample_rate.value())))))
        self.spin_averaging.setValue(self.settings.value("averaging", 1, int))
        self.settings.endGroup()

    def save_settings(self) -> None:
        self.settings.setValue("windowGeometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())

        self.settings.beginGroup("parameters")
        self.settings.setValue("mode", self.combo_mode.value())
        self.settings.setValue("triggerChannel", self.combo_trigger_channel.currentText())
        self.settings.setValue("triggerLevel", self.spin_trigger_level.value())
        self.settings.setValue("triggerEdge", self.combo_trigger_edge.value())
        self.settings.setValue("sampleRate", self.spin_sample_rate.value())
        self.settings.setValue("timeSpan", self.spin_time_span.value())
        self.settings.setValue("timeShift", self.spin_time_shift.value())
        self.settings.setValue("averaging", self.spin_averaging.value())
        self.settings.endGroup()

        self.settings.sync()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.save_settings()
        event.accept()

    def save_data(self) -> None:
        if not any((line.xData is not None and line.xData.shape[0]) for line in self.lines):
            QMessageBox.warning(self, self.tr("No Data"), self.tr("No data to save"))
            return
        self.settings.beginGroup("location")
        fn, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save Data"),
            self.settings.value("saveDirectory", "", str),
            self.tr("CSV File") + "(*.csv)",
        )
        self.settings.endGroup()
        if not fn:
            return
        self.setDisabled(True)
        datasets: dict[int, tuple[F64Array, F64Array] | tuple[None, None]] = {
            i: line.getOriginalDataset() for i, line in enumerate(self.lines, start=1) if line.isVisible()
        }
        datasets = {i: d for i, d in datasets.items() if not any(_d is None for _d in d)}
        if datasets:
            np.savetxt(
                fn,
                np.column_stack([next(d[0] for d in datasets.values())] + [d[1] for d in datasets.values()]),
                delimiter="\t",
                header="\t".join(("t", *(f"ch{i}" for i in datasets))),
            )
        else:
            QMessageBox.warning(
                self,
                self.tr("No Data to Save"),
                self.tr("There is no visible data. Enable a line or wait for data to come."),
            )
        self.setEnabled(True)
        self.settings.beginGroup("location")
        self.settings.setValue("saveDirectory", str(Path(fn).parent))
        self.settings.endGroup()

    @Slot()
    def on_button_start_clicked(self) -> None:
        self.button_start.setDisabled(True)
        self.button_single.setDisabled(True)
        self.spin_sample_rate.setDisabled(True)
        self.button_stop.setEnabled(True)
        for line in self.lines:
            line.setData([], [])
        self.mode = GUI.Mode.Continuous

    @Slot()
    def on_button_single_clicked(self) -> None:
        self.button_start.setDisabled(True)
        self.button_single.setDisabled(True)
        self.spin_sample_rate.setDisabled(True)
        self.button_stop.setEnabled(True)
        for line in self.lines:
            line.setData([], [])
        self.mode = GUI.Mode.Single

    @Slot()
    def on_button_stop_clicked(self) -> None:
        self.button_stop.setDisabled(True)
        self.spin_sample_rate.setEnabled(True)
        self.button_start.setEnabled(True)
        self.button_single.setEnabled(True)


@final
class App(GUI):
    def __init__(self) -> None:
        super().__init__()

        self.timer: QTimer = QTimer(self)
        self.timer.timeout.connect(self.on_timeout)
        self.timer.setInterval(1)

        self.results_queue: Queue[tuple[float, NDArray[np.float64]]] = Queue()
        self.measurement: NoiseMeasurement | None = None

        self.v: F64Array = np.empty((self.combo_trigger_channel.count(), 0))
        self.v_to_average: deque[F64Array] = deque(maxlen=self.spin_averaging.value())

        self.spin_time_shift.valueChanged.connect(self.on_spin_time_shift_value_changed)
        self.spin_time_span.valueChanged.connect(self.on_spin_time_span_value_changed)
        self.spin_sample_rate.valueChanged.connect(self.on_spin_sample_rate_value_changed)
        self.spin_averaging.valueChanged.connect(self.on_spin_averaging_value_changed)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.stop_measurement()
        return super().closeEvent(event)

    @Slot(float)
    @Slot(object)
    def on_spin_time_shift_value_changed(self, _value: float) -> None:
        self.v_to_average.clear()
        if self.measurement is None:
            self._plot()

    @Slot(float)
    @Slot(object)
    def on_spin_time_span_value_changed(self, value: float) -> None:
        self.spin_time_shift.setRange(-value / 2, value / 2)
        self.v_to_average.clear()
        if self.measurement is None:
            self._plot()

    @Slot(float)
    @Slot(object)
    def on_spin_sample_rate_value_changed(self, value: float) -> None:
        self.spin_time_span.setSingleStep(1.0 / value)
        self.spin_time_span.setMinimum(2.0 / value)
        self.spin_time_shift.setSingleStep(1.0 / value)
        self.spin_time_span.setDecimals(max(0, int(np.ceil(np.log10(value)))))
        self.spin_time_shift.setDecimals(max(0, int(np.ceil(np.log10(value)))))
        self.v_to_average.clear()

    @Slot(int)
    def on_spin_averaging_value_changed(self, value: int) -> None:
        self.v_to_average = deque(maxlen=value)

    @Slot()
    def on_button_start_clicked(self) -> None:
        self.v = np.empty((self.combo_trigger_channel.count(), 0))
        self.v_to_average.clear()
        super().on_button_start_clicked()
        self.timer.start()
        self.measurement = NoiseMeasurement(
            self.results_queue,
            *self.combo_trigger_channel.items().values(),
            sample_rate=self.spin_sample_rate.value(),
        )
        self.measurement.start()

    @Slot()
    def on_button_single_clicked(self) -> None:
        self.v = np.empty((self.combo_trigger_channel.count(), 0))
        self.v_to_average.clear()
        super().on_button_single_clicked()
        self.timer.start()
        self.measurement = NoiseMeasurement(
            self.results_queue,
            *self.combo_trigger_channel.items().values(),
            sample_rate=self.spin_sample_rate.value(),
        )
        self.measurement.start()

    def stop_measurement(self) -> None:
        if silent_alive(self.measurement):
            self.measurement.stop()
        clear_queue_after_process(self.measurement, self.results_queue)
        self.measurement = None
        self.timer.stop()

    @Slot()
    def on_button_stop_clicked(self) -> None:
        self.stop_measurement()
        super().on_button_stop_clicked()

    @Slot()
    def on_timeout(self) -> None:
        v: F64Array
        sample_rate: float = np.nan
        points_to_display: int
        while not self.results_queue.empty():
            try:
                sample_rate, v = self.results_queue.get_nowait()
            except Empty:
                break
            else:
                points_to_display = round(3 * sample_rate * self.spin_time_span.value())
                self.v = np.hstack((self.v, v))[:, -points_to_display:]

        if not np.isnan(sample_rate):
            self.spin_sample_rate.blockSignals(True)
            self.spin_sample_rate.setValue(sample_rate)
            self.spin_sample_rate.blockSignals(False)
            self._plot()

    def _plot(self) -> None:
        sample_rate: float = self.spin_sample_rate.value()
        trigger_level: float = self.spin_trigger_level.value()
        trigger_channel_index: int = self.combo_trigger_channel.currentIndex()
        v: F64Array = self.v.copy()
        trigger_channel_trend: F64Array = v[trigger_channel_index]
        triggers: I64Array
        match self.combo_trigger_edge.value():
            case "rising":
                triggers = np.argwhere(
                    (trigger_channel_trend[:-1] <= trigger_level) & (trigger_channel_trend[1:] >= trigger_level),
                )
            case "falling":
                triggers = np.argwhere(
                    (trigger_channel_trend[:-1] >= trigger_level) & (trigger_channel_trend[1:] <= trigger_level),
                )
            case "any":
                triggers = np.argwhere(
                    ((trigger_channel_trend[:-1] <= trigger_level) & (trigger_channel_trend[1:] >= trigger_level))
                    | ((trigger_channel_trend[:-1] >= trigger_level) & (trigger_channel_trend[1:] <= trigger_level)),
                )
            case _ as edge:
                raise ValueError(f"Invalid edge value: {edge}")
        mode: str = self.combo_mode.value()
        if not triggers.shape[0] and mode == "normal":
            return
        time_span: float = self.spin_time_span.value()
        time_shift: float = self.spin_time_shift.value()
        start: float = -time_span / 2 + time_shift
        stop: float = time_span / 2 + time_shift
        start_point: int = round(start * sample_rate)
        stop_point: int = round(stop * sample_rate)
        t: F64Array = np.linspace(start, stop, num=(stop_point - start_point), dtype=np.float64, endpoint=False)
        done: bool = False
        _v: F64Array
        __v: F64Array
        if mode == "normal" and triggers.shape[0]:
            trigger_point_index: int = triggers.shape[0] - 1
            trigger_point: int = int(triggers[trigger_point_index, 0])
            while trigger_point + t.shape[0] >= trigger_channel_trend.shape[0] and trigger_point_index >= 0:
                trigger_point_index -= 1
                trigger_point = int(triggers[trigger_point_index, 0])
            if (
                t.shape[0] <= v.shape[1]
                and -start_point < trigger_point
                and t.shape[0] + trigger_point < v.shape[1] > 0
            ):
                self.v_to_average.append(v[:, trigger_point + start_point : trigger_point + stop_point])
                done = len(self.v_to_average) >= self.spin_averaging.value()
            if len(self.v_to_average) > 1 and all_equally_shaped(self.v_to_average):
                __v = np.mean(self.v_to_average, axis=0)
            else:
                __v = v[:, trigger_point + start_point : trigger_point + stop_point]
            for index, line in enumerate(self.lines):
                _v = __v[index]
                if t.shape[0]:
                    if t.shape[0] > _v.shape[0]:
                        t = t[: _v.shape[0]]
                    line.setData(t, _v)
        elif mode == "auto":
            if t.shape[0] and v.shape[1] >= t.shape[0]:
                self.v_to_average.append(v[:, -t.shape[0] :])
                done = len(self.v_to_average) >= self.spin_averaging.value()
            if len(self.v_to_average) > 1 and all_equally_shaped(self.v_to_average):
                __v = np.mean(self.v_to_average, axis=0)
            else:
                __v = v
            for index, line in enumerate(self.lines):
                if t.shape[0]:
                    _v = v[index, -t.shape[0] :]
                    if t.shape[0] > _v.shape[0]:
                        t = t[: _v.shape[0]]
                    line.setData(t, _v)

        if done and self.mode == App.Mode.Single:
            self.stop_measurement()
            self.button_stop.click()


if __name__ == "__main__":
    app: QApplication = QApplication(sys.argv)
    if QT5:
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    window: App = App()
    window.show()
    app.exec()
