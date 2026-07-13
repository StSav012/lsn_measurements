import os.path
import sys
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import suppress
from datetime import datetime
from functools import partial
from itertools import zip_longest
from os import PathLike
from pathlib import Path
from queue import Empty, Queue
from string import Template
from threading import Event, Thread
from time import sleep
from typing import Any, Final, Literal, final

import numpy as np
import pyqtgraph as pg
from nidaqmx.constants import AcquisitionType
from nidaqmx.errors import DaqReadError
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.system.physical_channel import PhysicalChannel
from nidaqmx.task import Task
from numpy.typing import NDArray
from qtpy import QT5
from qtpy.QtCore import QSettings, QTimer, Qt, Slot
from qtpy.QtGui import QClipboard, QCloseEvent, QColor, QIcon, QKeySequence
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStyle,
    QTabWidget,
    QToolButton,
    QWidget,
)

from hardware import DIVIDER, DIVIDER_RESISTANCE, R, device_adc, device_dac, device_dio, offsets
from hardware.rigol import DG1000Z
from utils import all_equally_shaped, drain_queue
from utils.ni import measure_offsets

type F64Array = NDArray[np.float64]
type I64Array = NDArray[np.int64]


_MAX_ADC_SAMPLE_RATE: Final[float] = device_adc.ai_max_multi_chan_rate
color: QColor = QColor(Qt.GlobalColor.blue).lighter()


def return_none(c: Callable[[], Any]) -> None:
    c()


class NoiseMeasurement(Thread):
    def __init__(
        self,
        *channels: PhysicalChannel,
        sample_rate: float,
        source_channel: PhysicalChannel | None = None,
        source_voltage: float = 0.0,
        aux_channel: PhysicalChannel | None = None,
        aux_voltage: float = 0.0,
        delay_after_source_set: float = 0.0,
    ) -> None:
        super().__init__()
        self.results_queue: Queue[tuple[float, NDArray[np.float64]]] = Queue()

        self.channels: Sequence[PhysicalChannel] = channels
        self.sample_rate: Final[float] = sample_rate
        self.source_channel: Final[PhysicalChannel | None] = source_channel
        self.source_voltage: float = source_voltage
        self.aux_channel: Final[PhysicalChannel | None] = aux_channel
        self.aux_voltage: float = aux_voltage
        self.delay_after_source_set: float = delay_after_source_set

        self._done: Event = Event()

    def stop(self) -> None:
        self._done.set()

    def run(self) -> None:
        self._done.clear()

        task_adc: Task
        task_dac: Task
        with Task() as task_adc, Task() as task_dac:
            if self.source_channel is not None or self.aux_channel is not None:
                outs = {}
                vs = []
                if self.source_channel is not None:
                    outs[self.source_channel.name] = self.source_voltage
                if self.aux_channel is not None:
                    outs[self.aux_channel.name] = self.aux_voltage
                if outs:
                    for ch in sorted(outs):
                        task_dac.ao_channels.add_ao_voltage_chan(ch)
                        vs.append(outs[ch])
                    task_dac.write(vs)
                    task_dac.wait_until_done()
                    task_dac.stop()
                    self._done.wait(self.delay_after_source_set)

            for channel in self.channels:
                task_adc.ai_channels.add_ai_voltage_chan(channel.name)

            task_adc.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
            )

            adc_stream: AnalogMultiChannelReader = AnalogMultiChannelReader(task_adc.in_stream)

            number_of_channels: int = task_adc.number_of_channels
            sample_clock_rate: float = task_adc.timing.samp_clk_rate

            def reading_task_callback(
                _task_idx: int,
                _event_type: int,
                num_samples: int,
                _callback_data: object,
            ) -> Literal[0]:
                data: NDArray[np.float64] = np.empty((number_of_channels, num_samples), dtype=np.float64)
                with suppress(DaqReadError):
                    adc_stream.read_many_sample(data, num_samples)
                    self.results_queue.put((sample_clock_rate, data))
                return 0

            # noinspection PyTypeChecker
            task_adc.register_every_n_samples_acquired_into_buffer_event(
                task_adc.timing.samp_quant_samp_per_chan,
                reading_task_callback,
            )

            task_adc.start()

            self._done.wait()

            task_adc.stop()

            if outs:
                task_dac.write([0.0] * len(outs))
                task_dac.wait_until_done()
                task_dac.stop()

        drain_queue(self.results_queue)
        self._done.set()


class GUI(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        self.settings: QSettings = QSettings("SavSoft", "Oscilloscope", self)

        self.setWindowTitle(self.tr("Laser & SIN"))
        self.setWindowIcon(QIcon("lsn.svg"))

        self.figures: QTabWidget = QTabWidget(self)

        self.figure_dc: pg.PlotWidget = pg.PlotWidget(self)
        self.canvas_dc: pg.PlotItem = self.figure_dc.plotItem
        self.line_dc_without_averaging: pg.PlotDataItem = self.canvas_dc.plot(np.empty(0), pen="gray")
        self.line_dc: pg.PlotDataItem = self.canvas_dc.plot(np.empty(0), pen=color)

        self.figure_ac: pg.PlotWidget = pg.PlotWidget(self)
        self.canvas_ac: pg.PlotItem = self.figure_ac.plotItem
        self.line_ac_without_averaging: pg.PlotDataItem = self.canvas_ac.plot(np.empty(0), pen="gray")
        self.line_ac: pg.PlotDataItem = self.canvas_ac.plot(np.empty(0), pen=color)

        self.menu_bar: QMenuBar = QMenuBar(self)
        self.menu_file: QMenu = self.menu_bar.addMenu(self.tr("&File"))
        self.menu_view: QMenu = self.menu_bar.addMenu(self.tr("&View"))
        self.menu_help: QMenu = self.menu_bar.addMenu(self.tr("&Help"))

        self.source_box: QDockWidget = QDockWidget(self)
        self.combo_source_channel: pg.ComboBox = pg.ComboBox(self.source_box)
        self.combo_aux_channel: pg.ComboBox = pg.ComboBox(self.source_box)
        self.spin_source_voltage: pg.SpinBox = pg.SpinBox(self.source_box)
        self.spin_source_ballast_resistance: pg.SpinBox = pg.SpinBox(self.source_box)
        self.spin_source_divider_resistance: pg.SpinBox = pg.SpinBox(self.source_box)
        self.spin_source_current_divider: QDoubleSpinBox = QDoubleSpinBox(self.source_box)
        self.spin_source_current: pg.SpinBox = pg.SpinBox(self.source_box)
        self.spin_aux_voltage: pg.SpinBox = pg.SpinBox(self.source_box)
        self.check_source_current_step: QCheckBox = QCheckBox(self.tr("Auto increase by"), self.source_box)
        self.spin_source_current_step: pg.SpinBox = pg.SpinBox(self.source_box)
        self.spin_source_delay: pg.SpinBox = pg.SpinBox(self.source_box)
        self.source_box.setObjectName("source_box")

        self.parameters_box: QDockWidget = QDockWidget(self)
        self.parameters_box.setObjectName("parameters_box")
        self.combo_data_channel: pg.ComboBox = pg.ComboBox(self.parameters_box)
        self.combo_trigger_channel: pg.ComboBox = pg.ComboBox(self.parameters_box)
        self.spin_trigger_level: pg.SpinBox = pg.SpinBox(self.parameters_box)
        self.combo_trigger_edge: pg.ComboBox = pg.ComboBox(self.parameters_box)
        self.spin_sample_rate: pg.SpinBox = pg.SpinBox(self.parameters_box)
        self.spin_time_span: pg.SpinBox = pg.SpinBox(self.parameters_box)
        self.spin_averaging: QSpinBox = QSpinBox(self.parameters_box)
        self.spin_g_voltage: pg.SpinBox = pg.SpinBox(self.parameters_box)
        self.spin_g_pulse_width: pg.SpinBox = pg.SpinBox(self.parameters_box)

        self.stats_box: QDockWidget = QDockWidget(self)
        self.stats_box.setObjectName("stats_box")
        self.label_stats_noise: pg.ValueLabel = pg.ValueLabel(self.stats_box, siPrefix=True)
        self.label_stats_magnitude: pg.ValueLabel = pg.ValueLabel(self.stats_box, siPrefix=True)
        self.label_stats_area: pg.ValueLabel = pg.ValueLabel(self.stats_box, siPrefix=True)

        self.saving_box: QDockWidget = QDockWidget(self)
        self.saving_box.setObjectName("saving_box")
        self.check_saving: QCheckBox = QCheckBox(self.tr("Auto save as"), self.saving_box)
        self.path_saving: QLineEdit = QLineEdit(self.saving_box)
        self.button_copy_saving_path: QToolButton = QToolButton(self.saving_box)

        self.buttons_box: QDockWidget = QDockWidget(self)
        self.buttons_box.setObjectName("buttons_box")
        self.button_start: QPushButton = QPushButton(self.buttons_box)
        self.button_stop: QPushButton = QPushButton(self.buttons_box)

        self.setup_ui_appearance()
        self.load_settings()
        self.setup_actions()

    def setup_ui_appearance(self) -> None:
        self.canvas_dc.vb.menu.addAction(self.tr("Clea&r"), self.on_canvas_clear_triggered)
        self.canvas_dc.vb.menu.addAction(self.tr("Copy &Data"), self.copy_data)

        self.canvas_ac.vb.menu.addAction(self.tr("Clea&r"), self.on_canvas_clear_triggered)
        self.canvas_ac.vb.menu.addAction(self.tr("Copy &Data"), self.copy_data)

        self.combo_source_channel.setItems({ch.name: ch for ch in device_dac.ao_physical_chans})
        self.combo_aux_channel.setItems({ch.name: ch for ch in device_dac.ao_physical_chans})
        ch_data: dict[str, PhysicalChannel] = {ch.name: ch for ch in device_adc.ai_physical_chans}
        self.combo_data_channel.setItems(ch_data)
        self.combo_trigger_channel.setItems(ch_data)
        self.combo_trigger_edge.setItems(
            {self.tr("Rising"): "rising", self.tr("Falling"): "falling", self.tr("Any"): "any"},
        )

        opts: dict[str, bool | str | int | float]
        opts = dict(
            suffix=self.tr("V"),
            siPrefix=True,
            decimals=3,
            dec=True,
            compactHeight=False,
            format="{scaledValue:.{decimals}f}{suffixGap}{siPrefix}{suffix}",
            scaleAtZero=1.0,
        )
        self.spin_source_voltage.setOpts(**opts)
        self.spin_source_voltage.setRange(-10.0, 10.0)
        self.spin_trigger_level.setOpts(**opts)
        self.spin_g_voltage.setOpts(**opts)
        self.spin_trigger_level.setRange(min(device_adc.ai_voltage_rngs), max(device_adc.ai_voltage_rngs))
        self.spin_g_voltage.setRange(0.0, 10.0)
        opts.update(
            dict(
                suffix=self.tr("S/s"),
                format="{scaledValue:.{decimals}f}{suffixGap}{siPrefix}{suffix}",
            )
        )
        self.spin_sample_rate.setOpts(**opts)
        self.spin_sample_rate.setRange(device_adc.ai_min_rate, _MAX_ADC_SAMPLE_RATE)
        opts.update(
            dict(
                suffix=self.tr("s"),
                format="{scaledValue:.{decimals}f}{suffixGap}{siPrefix}{suffix}",
                scaleAtZero=1.0,
            )
        )
        self.spin_time_span.setOpts(**opts)
        self.spin_source_delay.setOpts(**opts)
        self.spin_source_delay.setMinimum(0.0)
        opts.update(
            dict(
                decimals=2,
                scaleAtZero=0.001,
            )
        )
        self.spin_g_pulse_width.setOpts(**opts)
        self.spin_g_pulse_width.setMinimum(20e-9)
        opts.update(
            dict(
                suffix=self.tr("Ω"),
                decimals=3,
                format="{scaledValue:.{decimals}f}{suffixGap}{siPrefix}{suffix}",
            )
        )
        self.spin_source_ballast_resistance.setOpts(**opts)
        self.spin_source_divider_resistance.setOpts(**opts)
        self.spin_source_current_divider.setDecimals(3)
        opts.update(
            dict(
                suffix=self.tr("A"),
                format="{scaledValue:.{decimals}f}{suffixGap}{siPrefix}{suffix}",
                scaleAtZero=1e-9,
            )
        )
        self.spin_source_current.setOpts(**opts)
        self.spin_source_current_step.setOpts(**opts)
        opts.update(
            dict(
                suffix=self.tr("V"),
                format="{scaledValue:.{decimals}f}{suffixGap}{siPrefix}{suffix}",
                scaleAtZero=1.0,
            )
        )
        self.spin_aux_voltage.setOpts(**opts)

        self.spin_time_span.setMinimum(2.0 / _MAX_ADC_SAMPLE_RATE)
        self.spin_time_span.setMaximum(np.inf)
        self.spin_time_span.setSingleStep(1.0 / _MAX_ADC_SAMPLE_RATE)
        self.spin_time_span.setDecimals(max(0, int(np.ceil(np.log10(_MAX_ADC_SAMPLE_RATE)))))

        self.spin_averaging.setRange(1, 50000)

        self.label_stats_noise.suffix = self.tr("V")
        self.label_stats_magnitude.suffix = self.tr("V")
        self.label_stats_area.suffix = self.tr("V×s")
        self.label_stats_noise.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.label_stats_magnitude.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.label_stats_area.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)

        self.figure_dc.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.canvas_dc.setLabels(
            left=(self.tr("DC Voltage"), self.tr("V")),
            bottom=(self.tr("Time"), self.tr("s")),
        )
        self.canvas_dc.showGrid(x=True, y=True)

        self.figure_ac.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.canvas_ac.setLabels(
            left=(self.tr("AC Voltage"), self.tr("V")),
            bottom=(self.tr("Time"), self.tr("s")),
        )
        self.canvas_ac.showGrid(x=True, y=True)

        self.menu_file.addAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
            self.tr("&Save As…"),
            partial(self.save_data, all_lines=True),
            QKeySequence.StandardKey.Save,
        )
        self.menu_file.addAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
            self.tr("Save Last Line As…"),
            partial(self.save_data, all_lines=False),
            QKeySequence.StandardKey.SaveAs,
        )
        self.menu_file.addSeparator()
        self.menu_file.addAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCloseButton),
            self.tr("&Quit"),
            lambda: return_none(self.close),
        )
        self.menu_view.addAction(self.source_box.toggleViewAction())
        self.menu_view.addAction(self.parameters_box.toggleViewAction())
        self.menu_view.addAction(self.saving_box.toggleViewAction())
        self.menu_view.addAction(self.buttons_box.toggleViewAction())
        self.menu_help.addAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarMenuButton),
            self.tr("About &Qt…"),
            partial(QMessageBox.aboutQt, self),
        )
        self.setMenuBar(self.menu_bar)

        source_layout: QFormLayout = QFormLayout()
        source_layout.addRow(self.tr("Bias source:"), self.combo_source_channel)
        source_layout.addRow(self.tr("Aux source:"), self.combo_aux_channel)
        source_layout.addRow(self.tr("Voltage:"), self.spin_source_voltage)
        source_layout.addRow(self.tr("Ballast resistance:"), self.spin_source_ballast_resistance)
        source_layout.addRow(self.tr("Divider resistance:"), self.spin_source_divider_resistance)
        source_layout.addRow(self.tr("Current divider:"), self.spin_source_current_divider)
        source_layout.addRow(self.tr("Current:"), self.spin_source_current)
        source_layout.addRow(self.tr("Aux voltage:"), self.spin_aux_voltage)
        source_layout.addRow(self.check_source_current_step, self.spin_source_current_step)
        source_layout.addRow(self.tr("Delay after source set:"), self.spin_source_delay)
        source_box_widget: QWidget = QWidget(self.source_box)
        source_box_widget.setLayout(source_layout)
        self.source_box.setWidget(source_box_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.source_box)
        self.source_box.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.source_box.setWindowTitle(self.tr("Source"))

        parameters_layout: QFormLayout = QFormLayout()
        parameters_layout.addRow(self.tr("Data channel:"), self.combo_data_channel)
        parameters_layout.addRow(self.tr("Trigger channel:"), self.combo_trigger_channel)
        parameters_layout.addRow(self.tr("Trigger level:"), self.spin_trigger_level)
        parameters_layout.addRow(self.tr("Trigger edge:"), self.combo_trigger_edge)
        parameters_layout.addRow(self.tr("Sample rate:"), self.spin_sample_rate)
        parameters_layout.addRow(self.tr("Time span:"), self.spin_time_span)
        parameters_layout.addRow(self.tr("Averaging:"), self.spin_averaging)
        parameters_layout.addRow(self.tr("Gen voltage:"), self.spin_g_voltage)
        parameters_layout.addRow(self.tr("Gen pulse width:"), self.spin_g_pulse_width)
        parameters_box_widget: QWidget = QWidget(self.parameters_box)
        parameters_box_widget.setLayout(parameters_layout)
        self.parameters_box.setWidget(parameters_box_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.parameters_box)
        self.parameters_box.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.parameters_box.setWindowTitle(self.tr("Parameters"))

        stats_layout: QFormLayout = QFormLayout()
        stats_layout.addRow(self.tr("Noise:"), self.label_stats_noise)
        stats_layout.addRow(self.tr("Magnitude:"), self.label_stats_magnitude)
        stats_layout.addRow(self.tr("Area:"), self.label_stats_area)
        stats_box_widget: QWidget = QWidget(self.stats_box)
        stats_box_widget.setLayout(stats_layout)
        self.stats_box.setWidget(stats_box_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.stats_box)
        self.stats_box.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.stats_box.setWindowTitle(self.tr("Statistics"))

        saving_layout: QHBoxLayout = QHBoxLayout()
        self.path_saving.setToolTip("""Use following placeholders:
$source_current
$source_voltage
$pulse_width
$pulse_voltage
$pulse_period
$averaging
$sample_rate
$now
$today
$time""")
        self.button_copy_saving_path.setText(self.tr("&Copy"))
        saving_layout.addWidget(self.check_saving, 0)
        saving_layout.addWidget(self.path_saving, 1)
        saving_layout.addWidget(self.button_copy_saving_path, 0)
        saving_box_widget: QWidget = QWidget(self.saving_box)
        saving_box_widget.setLayout(saving_layout)
        self.saving_box.setWidget(saving_box_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.saving_box)
        self.saving_box.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.saving_box.setWindowTitle(self.tr("Saving"))

        buttons_layout: QHBoxLayout = QHBoxLayout()
        buttons_layout.addWidget(self.button_start)
        buttons_layout.addWidget(self.button_stop)
        buttons_box_widget: QWidget = QWidget(self.buttons_box)
        buttons_box_widget.setLayout(buttons_layout)
        self.buttons_box.setWidget(buttons_box_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.buttons_box)
        self.buttons_box.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.buttons_box.setWindowTitle(self.tr("Controls"))

        self.button_start.setText(self.tr("Start"))
        self.button_stop.setText(self.tr("Stop"))
        self.button_stop.setDisabled(True)

        self.figures.addTab(self.figure_dc, self.tr("DC"))
        self.figures.addTab(self.figure_ac, self.tr("AC"))
        self.setCentralWidget(self.figures)

    def setup_actions(self) -> None:
        self.spin_source_voltage.valueChanged.connect(self.on_spin_source_voltage_value_changed)
        self.spin_source_current_divider.valueChanged.connect(self.on_spin_source_current_divider_value_changed)
        self.spin_source_divider_resistance.valueChanged.connect(self.on_spin_source_divider_resistance_value_changed)
        self.spin_source_ballast_resistance.valueChanged.connect(self.on_spin_source_ballast_resistance_value_changed)
        self.spin_source_current.valueChanged.connect(self.on_spin_source_current_value_changed)
        self.spin_aux_voltage.valueChanged.connect(self.on_spin_aux_voltage_value_changed)
        self.button_start.clicked.connect(self.on_button_start_clicked)
        self.button_stop.clicked.connect(self.on_button_stop_clicked)
        self.button_copy_saving_path.clicked.connect(self.on_button_copy_saving_path_clicked)

    def load_settings(self) -> None:
        self.restoreGeometry(self.settings.value("windowGeometry", b""))
        self.restoreState(self.settings.value("windowState", b""))

        self.settings.beginGroup("parameters")
        with suppress(ValueError):
            # `ValueError` might occur when there is no such channel present
            self.combo_data_channel.setText(
                self.settings.value("dataChannel", self.combo_data_channel.currentText(), str),
            )
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
        self.spin_time_span.setSingleStep(1.0 / self.spin_sample_rate.value())
        self.spin_time_span.setMinimum(2.0 / self.spin_sample_rate.value())
        self.spin_time_span.setDecimals(max(0, int(np.ceil(np.log10(self.spin_sample_rate.value())))))
        self.spin_averaging.setValue(self.settings.value("averaging", 1, int))
        with suppress(ValueError):
            # `ValueError` might occur when there is no such channel present
            self.combo_source_channel.setText(
                self.settings.value("sourceChannel", self.combo_source_channel.currentText(), str),
            )
        with suppress(ValueError):
            # `ValueError` might occur when there is no such channel present
            self.combo_aux_channel.setText(
                self.settings.value("auxChannel", self.combo_aux_channel.currentText(), str),
            )
        self.spin_source_voltage.setValue(self.settings.value("sourceVoltage", 0.0, float))
        self.spin_source_ballast_resistance.setValue(self.settings.value("sourceBallastResistance", R, float))
        self.spin_source_divider_resistance.setValue(
            self.settings.value("sourceDividerResistance", DIVIDER_RESISTANCE, float),
        )
        self.spin_source_current_divider.setValue(self.settings.value("sourceCurrentDivider", DIVIDER, float))
        self.spin_source_current_step.setValue(self.settings.value("sourceCurrentStep", 0.0, float))
        self.spin_source_delay.setValue(self.settings.value("sourceDelay", 0.0, float))
        self.spin_source_current.blockSignals(True)
        self.spin_source_current.setValue(
            self.spin_source_voltage.value()
            / (
                self.spin_source_current_divider.value()
                * (self.spin_source_divider_resistance.value() + self.spin_source_ballast_resistance.value())
            ),
        )
        self.spin_source_current.blockSignals(False)
        self.spin_aux_voltage.setValue(self.settings.value("auxVoltage", 0.0, float))
        self.spin_g_voltage.setValue(self.settings.value("genVoltage", 3.0, float))
        self.spin_g_pulse_width.setValue(self.settings.value("genPulseWidth", 1e-6, float))

        self.path_saving.setText(
            self.settings.value(
                "savingTemplateForLaser",
                os.path.join(
                    self.settings.value("saveDirectory", "", str),
                    "bias = $source_current, "
                    "bias = $source_voltage, "
                    "pulse width = $pulse_width, "
                    "pulse height = $pulse_voltage, "
                    "pulse period = $pulse_period, "
                    "averaging = $averaging"
                    ".csv",
                ),
                str,
            ),
        )
        self.settings.endGroup()

    def save_settings(self) -> None:
        self.settings.setValue("windowGeometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())

        self.settings.beginGroup("parameters")
        self.settings.setValue("dataChannel", self.combo_data_channel.currentText())
        self.settings.setValue("triggerChannel", self.combo_trigger_channel.currentText())
        self.settings.setValue("triggerLevel", self.spin_trigger_level.value())
        self.settings.setValue("triggerEdge", self.combo_trigger_edge.value())
        self.settings.setValue("sampleRate", self.spin_sample_rate.value())
        self.settings.setValue("timeSpan", self.spin_time_span.value())
        self.settings.setValue("averaging", self.spin_averaging.value())
        self.settings.setValue("sourceChannel", self.combo_source_channel.currentText())
        self.settings.setValue("auxChannel", self.combo_aux_channel.currentText())
        self.settings.setValue("sourceVoltage", self.spin_source_voltage.value())
        self.settings.setValue("auxVoltage", self.spin_aux_voltage.value())
        self.settings.setValue("sourceBallastResistance", self.spin_source_ballast_resistance.value())
        self.settings.setValue("sourceDividerResistance", self.spin_source_divider_resistance.value())
        self.settings.setValue("sourceCurrentDivider", self.spin_source_current_divider.value())
        self.settings.setValue("sourceCurrentStep", self.spin_source_current_step.value())
        self.settings.setValue("sourceDelay", self.spin_source_delay.value())

        self.settings.setValue("genVoltage", self.spin_g_voltage.value())
        self.settings.setValue("genPulseWidth", self.spin_g_pulse_width.value())

        self.settings.setValue("savingTemplateForLaser", self.path_saving.text())
        self.settings.endGroup()

        self.settings.sync()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.save_settings()
        event.accept()

    def copy_data(self, *, all_lines: bool = False) -> None:
        dataset: list[F64Array | None]
        if all_lines:
            dataset = [
                next(
                    (
                        _d[0]
                        for line in self.canvas_dc.listDataItems()
                        if (_d := line.getOriginalDataset()) is not None and line is not self.line_dc_without_averaging
                    ),
                    None,
                )
            ] + [
                _d[1]
                for line in self.canvas_dc.listDataItems()
                if (_d := line.getOriginalDataset())[1] is not None and line is not self.line_dc_without_averaging
            ]
        else:
            dataset = next(
                (
                    _d
                    for line in self.canvas_dc.listDataItems()
                    if (_d := line.getOriginalDataset())[0] is not None
                    and _d[1] is not None
                    and line is not self.line_dc_without_averaging
                ),
                (None, None),
            )
        if not all(_d is None for _d in dataset):
            lines: deque[str] = deque()
            for values in zip_longest(*dataset, fillvalue=""):
                lines.append("\t".join(map(str, values)))
            QApplication.clipboard().setText(os.linesep.join(lines), QClipboard.Mode.Clipboard)
        else:
            QMessageBox.warning(
                self,
                self.tr("No Data to Copy"),
                self.tr("Wait for data to come."),
            )

    def save_data(self, fn: str | PathLike[str] = "", *, all_lines: bool = False) -> None:
        dataset: list[F64Array] | list[None] = list(self.line_dc.getOriginalDataset())
        if any(_d is None for _d in dataset):
            QMessageBox.warning(self, self.tr("No Data"), self.tr("No data to save."))
            return
        if not fn:
            self.settings.beginGroup("location")
            fn, _ = QFileDialog.getSaveFileName(
                self,
                self.tr("Save Data"),
                os.path.join(
                    self.settings.value("saveDirectory", "", str),
                    f"bias = {self.spin_source_current.text()}, "
                    f"bias = {self.spin_source_voltage.text()}, " * (not all_lines)
                    + f"pulse width = {self.spin_g_pulse_width.text()}, "
                    f"pulse height = {self.spin_g_voltage.text()}, "
                    f"averaging = {self.spin_averaging.text()}"
                    ".csv",
                ),
                self.tr("CSV File (*.csv)"),
            )
            self.settings.endGroup()
        elif (
            Path(fn).exists()
            and QMessageBox.question(
                self,
                self.tr("Confirm Overwrite"),
                self.tr("File {} already exists. Overwrite?").format(str(fn)),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            == QMessageBox.StandardButton.No
        ):
            return
        if not fn:
            return
        fn = Path(fn)
        if not fn.parent.exists():
            fn.parent.mkdir(parents=True)

        self.setDisabled(True)

        datasets: dict[str, F64Array] = {self.tr("time"): dataset[0]}
        if all_lines:
            for index, line in enumerate(self.canvas_dc.listDataItems(), start=1):
                if line is self.line_dc_without_averaging:
                    continue
                _d = line.getOriginalDataset()
                if _d[0] is None:
                    continue
                for key in [line.name(), f"{self.combo_data_channel.currentText()} [{index}]"]:
                    if key and key not in datasets:
                        datasets[key] = _d[1]
                        break
        else:
            datasets[self.combo_data_channel.currentText()] = dataset[1]

        if not any(_d is None for _d in datasets.values()):
            # noinspection PyTypeChecker
            np.savetxt(
                fn,
                np.column_stack(list(datasets.values())),
                delimiter="\t",
                header="\t".join(datasets.keys()),
                comments="",
            )
        else:
            QMessageBox.warning(
                self,
                self.tr("No Data to Save"),
                self.tr("Wait for data to come."),
            )

        self.setEnabled(True)
        self.settings.beginGroup("location")
        self.settings.setValue("saveDirectory", str(fn.parent))
        self.settings.endGroup()

    @Slot()
    def on_canvas_clear_triggered(self) -> None:
        for line in self.canvas_dc.listDataItems():
            if line is self.line_dc or line is self.line_dc_without_averaging:
                line.setData([], [])
            else:
                self.canvas_dc.removeItem(line)
        self.canvas_dc.avgCurves.clear()
        for line in self.canvas_ac.listDataItems():
            if line is self.line_ac or line is self.line_ac_without_averaging:
                line.setData([], [])
            else:
                self.canvas_ac.removeItem(line)
        self.canvas_ac.avgCurves.clear()

    @Slot(object)
    def on_spin_source_voltage_value_changed(self, value: float) -> None:
        self.spin_source_current.blockSignals(True)
        self.spin_source_current.setValue(
            value
            / (
                self.spin_source_current_divider.value()
                * (self.spin_source_divider_resistance.value() + self.spin_source_ballast_resistance.value())
            ),
        )
        self.spin_source_current.blockSignals(False)

    @Slot(float)
    def on_spin_source_current_divider_value_changed(self, value: float) -> None:
        self.spin_source_current.blockSignals(True)
        self.spin_source_current.setValue(
            self.spin_source_voltage.value()
            / (value * (self.spin_source_divider_resistance.value() + self.spin_source_ballast_resistance.value())),
        )
        self.spin_source_current.blockSignals(False)

    @Slot(object)
    def on_spin_source_divider_resistance_value_changed(self, value: float) -> None:
        self.spin_source_current.blockSignals(True)
        self.spin_source_current.setValue(
            self.spin_source_voltage.value()
            / (self.spin_source_current_divider.value() * (value + self.spin_source_ballast_resistance.value())),
        )
        self.spin_source_current.blockSignals(False)

    @Slot(object)
    def on_spin_source_ballast_resistance_value_changed(self, value: float) -> None:
        self.spin_source_current.blockSignals(True)
        self.spin_source_current.setValue(
            self.spin_source_voltage.value()
            / (self.spin_source_current_divider.value() * (self.spin_source_divider_resistance.value() + value)),
        )
        self.spin_source_current.blockSignals(False)

    @Slot(object)
    def on_spin_source_current_value_changed(self, value: float) -> None:
        self.spin_source_voltage.blockSignals(True)
        self.spin_source_voltage.setValue(
            value
            * (
                self.spin_source_current_divider.value()
                * (self.spin_source_divider_resistance.value() + self.spin_source_ballast_resistance.value())
            ),
        )
        self.spin_source_voltage.blockSignals(False)

    @Slot(object)
    def on_spin_aux_voltage_value_changed(self, value: float) -> None:
        pass

    @Slot()
    def on_button_start_clicked(self) -> None:
        self.button_start.setDisabled(True)
        self.combo_data_channel.setDisabled(True)
        self.combo_trigger_channel.setDisabled(True)
        self.spin_sample_rate.setDisabled(True)
        self.button_stop.setEnabled(True)
        self.line_dc.setData([], [])
        self.line_dc_without_averaging.setData([], [])
        self.line_ac.setData([], [])
        self.line_ac_without_averaging.setData([], [])

    @Slot()
    def on_button_stop_clicked(self) -> None:
        self.button_stop.setDisabled(True)
        self.combo_data_channel.setEnabled(True)
        self.combo_trigger_channel.setEnabled(True)
        self.spin_sample_rate.setEnabled(True)
        self.spin_g_voltage.setEnabled(True)
        self.spin_g_pulse_width.setEnabled(True)
        self.button_start.setEnabled(True)

    @Slot()
    def on_button_copy_saving_path_clicked(self) -> None:
        now: datetime = datetime.now()
        QApplication.clipboard().setText(
            Template(self.path_saving.text()).safe_substitute(
                source_current=self.spin_source_current.text(),
                aux_voltage=self.spin_aux_voltage.text(),
                source_voltage=self.spin_source_voltage.text(),
                pulse_width=self.spin_g_pulse_width.text(),
                pulse_voltage=self.spin_g_voltage.text(),
                pulse_period=self.spin_time_span.text(),
                averaging=self.spin_averaging.text(),
                sample_rate=self.spin_sample_rate.text(),
                now=now.isoformat().replace(":", "-"),
                today=now.date().isoformat(),
                time=now.time().isoformat().replace(":", "-"),
            )
        )


@final
class App(GUI):
    def __init__(self) -> None:
        super().__init__()

        self.g: DG1000Z = DG1000Z(reset=True)

        self.timer: QTimer = QTimer(self)
        self.timer.timeout.connect(self.on_timeout)
        self.timer.setInterval(1)

        self.measurement: NoiseMeasurement | None = None

        self.v: F64Array = np.empty((0, 0))
        self.v_to_average: deque[F64Array] = deque(maxlen=self.spin_averaging.value())

        self.spin_g_pulse_width.valueChanged.connect(self.on_spin_g_pulse_width_value_changed)
        self.spin_time_span.valueChanged.connect(self.on_spin_time_span_value_changed)
        self.spin_sample_rate.valueChanged.connect(self.on_spin_sample_rate_value_changed)
        self.spin_averaging.valueChanged.connect(self.on_spin_averaging_value_changed)

        self.spin_g_pulse_width.setMinimum(self.spin_time_span.value() * 1e-5)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.stop_measurement()
        return super().closeEvent(event)

    @Slot(float)
    @Slot(object)
    def on_spin_g_pulse_width_value_changed(self, value: float) -> None:
        self.g.source1.function.pulse.width = value
        self.v_to_average.clear()
        if self.measurement is None:
            self._plot()

    @Slot(float)
    @Slot(object)
    def on_spin_time_span_value_changed(self, value: float) -> None:
        self.g.source1.function.pulse.period = value
        self.spin_g_pulse_width.setMinimum(value * 1e-5)
        self.v_to_average.clear()
        if self.measurement is None:
            self._plot()

    @Slot(float)
    @Slot(object)
    def on_spin_sample_rate_value_changed(self, value: float) -> None:
        self.spin_time_span.setSingleStep(1.0 / value)
        self.spin_time_span.setMinimum(2.0 / value)
        self.spin_time_span.setDecimals(max(0, int(np.ceil(np.log10(value)))))
        self.v_to_average.clear()

    @Slot(int)
    def on_spin_averaging_value_changed(self, value: int) -> None:
        self.v_to_average = deque(maxlen=value)

    @Slot()
    def on_button_start_clicked(self) -> None:
        super().on_button_start_clicked()
        if not offsets:
            measure_offsets(1.0)
        self.start_measurement()

    @Slot()
    def on_button_stop_clicked(self) -> None:
        self.stop_measurement()
        super().on_button_stop_clicked()

    def start_measurement(self) -> None:
        device_adc.reset_device()
        if device_adc.name != device_dac.name:
            device_dac.reset_device()
        if device_dio.name != device_adc.name and device_dio.name != device_dac.name:
            device_dio.reset_device()

        self.g.output1.state = False
        self.g.output1.load = "infinity"
        self.g.source1.function.shape = "pulse"
        self.g.source1.function.pulse.period = self.spin_time_span.value()
        self.g.source1.function.pulse.width = self.spin_g_pulse_width.value()
        self.g.source1.function.pulse.transition.both = "minimum"
        self.g.source1.voltage.high = 0.0
        self.g.source1.voltage.low = -self.spin_g_voltage.value()
        self.g.output1.sync.state = True
        sleep(1)  # give this mongol time to comprehend the prior commands
        self.g.output1.polarity = "inverted"
        self.g.output1.state = True
        while not self.g.output1.state:
            sleep(0.01)

        channels: list[PhysicalChannel]
        if self.combo_data_channel.currentIndex() > self.combo_trigger_channel.currentIndex():
            channels = [self.combo_trigger_channel.value(), self.combo_data_channel.value()]
        elif self.combo_data_channel.currentIndex() < self.combo_trigger_channel.currentIndex():
            channels = [self.combo_data_channel.value(), self.combo_trigger_channel.value()]
        else:
            channels = [self.combo_data_channel.value()]
        self.v = np.empty((len(channels), 0))
        self.v_to_average.clear()
        self.timer.start()
        self.measurement = NoiseMeasurement(
            *channels,
            sample_rate=self.spin_sample_rate.value(),
            source_channel=self.combo_source_channel.value(),
            source_voltage=self.spin_source_voltage.value(),
            aux_channel=self.combo_aux_channel.value(),
            aux_voltage=self.spin_aux_voltage.value(),
            delay_after_source_set=self.spin_source_delay.value(),
        )
        self.measurement.start()

    def stop_measurement(self) -> None:
        self.timer.stop()
        self.check_source_current_step.setChecked(False)
        self.g.output1.state = False
        if self.measurement is not None:
            self.measurement.stop()
            self.measurement.join()
            self.measurement = None
        self.v_to_average.clear()

        # dataset: tuple[F64Array, F64Array] | tuple[None, None] = self.line_dc.getOriginalDataset()
        # if any(_d is None for _d in dataset):
        #     return
        # std: float = np.std(dataset[1][: dataset[1].shape[0] // 2], dtype=float)
        # base: float = np.mean(dataset[1][: dataset[1].shape[0] // 2], dtype=float)
        # print("base voltage", base, sep=" = ")
        # print("noise std", std, sep=" = ")
        # print("amplitude", base - np.min(dataset[1][dataset[1].shape[0] // 2 :]) - 3.0 * std, sep=" = ")
        # print("3 x noise", 3.0 * std, sep=" = ")
        # print(
        #     "area",
        #     (np.mean(dataset[1][dataset[1].shape[0] // 2 :], dtype=float) - base)
        #     * (dataset[0][-1] - dataset[0][dataset[1].shape[0] // 2]),
        #     sep=" = ",
        # )

    @Slot()
    def on_timeout(self) -> None:
        v: F64Array
        sample_rate: float = np.nan
        while self.measurement is not None and not self.measurement.results_queue.empty():
            try:
                sample_rate, v = self.measurement.results_queue.get_nowait()
            except Empty:
                break
            else:
                self.v = np.hstack((self.v, v))

        if not np.isnan(sample_rate):
            self.spin_sample_rate.blockSignals(True)
            self.spin_sample_rate.setValue(sample_rate)
            self.spin_sample_rate.blockSignals(False)
            if self._plot():
                if self.check_saving.isChecked():
                    now: datetime = datetime.now()
                    path_parts: list[str] = list(
                        Path(
                            Template(self.path_saving.text()).safe_substitute(
                                source_current=self.spin_source_current.text(),
                                aux_voltage=self.spin_aux_voltage.text(),
                                source_voltage=self.spin_source_voltage.text(),
                                pulse_width=self.spin_g_pulse_width.text(),
                                pulse_voltage=self.spin_g_voltage.text(),
                                pulse_period=self.spin_time_span.text(),
                                averaging=self.spin_averaging.text(),
                                sample_rate=self.spin_sample_rate.text(),
                                now=now.isoformat().replace(":", "-"),
                                today=now.date().isoformat(),
                                time=now.time().isoformat().replace(":", "-"),
                            )
                        ).parts
                    )
                    for i, p in enumerate(path_parts.copy()):
                        if "*" in p or "?" in p:
                            part_variants: list[Path] = list(Path(*path_parts[:i]).glob(p))
                            if not part_variants:
                                path_parts[i] = p.replace("?", "").replace("*", "")
                            else:
                                path_parts[i] = part_variants[0].name
                    path: Path = Path(*path_parts)
                    self.save_data(
                        path,
                        all_lines=False,
                    )
                if self.check_source_current_step.isChecked():
                    if self.measurement is not None:
                        self.measurement.stop()
                        self.measurement.join()
                        self.measurement = None
                    self.timer.stop()

                    self.spin_source_current.setValue(
                        self.spin_source_current.value() + self.spin_source_current_step.value()
                    )
                    self.spin_source_voltage.blockSignals(True)
                    self.spin_source_voltage.setValue(
                        self.spin_source_current.value()
                        * (
                            self.spin_source_current_divider.value()
                            * (
                                self.spin_source_divider_resistance.value()
                                + self.spin_source_ballast_resistance.value()
                            )
                        ),
                    )
                    self.spin_source_voltage.blockSignals(False)
                    if -10.0 < self.spin_source_voltage.value() < 10.0:
                        import random

                        c: QColor = pg.functions.intColor(random.randint(0, 128), hues=32, values=4)
                        self.line_dc = self.canvas_dc.plot(
                            np.empty(0),
                            pen=c,
                        )
                        self.line_ac = self.canvas_ac.plot(
                            np.empty(0),
                            pen=c,
                        )
                        self.start_measurement()
                else:
                    self.stop_measurement()
                    super().on_button_stop_clicked()

    def _plot(self) -> bool:
        done: bool = False
        sample_rate: float = self.spin_sample_rate.value()
        trigger_level: float = self.spin_trigger_level.value()
        data_index: int = int(self.combo_trigger_channel.currentIndex() < self.combo_data_channel.currentIndex())
        trigger_channel_index: int = int(
            self.combo_trigger_channel.currentIndex() > self.combo_data_channel.currentIndex(),
        )
        v: F64Array = self.v
        if trigger_channel_index > v.shape[0] or data_index > v.shape[0]:
            return False
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
        if not triggers.shape[0]:
            return done
        time_span: float = self.spin_time_span.value()
        start: float = -time_span / 2
        stop: float = time_span / 2
        start_point: int = round(start * sample_rate)
        stop_point: int = round(stop * sample_rate)
        t: F64Array = np.linspace(start, stop, num=(stop_point - start_point), dtype=np.float64, endpoint=False)
        __v: F64Array
        trigger_point_index: int = triggers.shape[0] - 1
        trigger_point: int = int(triggers[trigger_point_index, 0])
        while trigger_point + t.shape[0] >= trigger_channel_trend.shape[0] and trigger_point_index >= 0:
            trigger_point_index -= 1
            trigger_point = int(triggers[trigger_point_index, 0])
        if trigger_point_index < 0:
            return done
        if (
            t.shape[0] <= v.shape[1]
            and trigger_point + start_point >= 0
            and trigger_point + stop_point < v.shape[1]
            and t.shape[0] + trigger_point < v.shape[1] > 0
        ):
            __v = v[data_index, trigger_point + start_point : trigger_point + stop_point]
            self.v = self.v[:, trigger_point + stop_point :]
            self.v_to_average.append(__v)
            if t.shape[0]:
                __t = t
                if __t.shape[0] > __v.shape[0]:
                    __t = __t[: __v.shape[0]]
                self.line_dc_without_averaging.setData(__t, __v)
                self.line_ac_without_averaging.setData(__t, __v - np.nanmedian(__v))
        if self.v_to_average and all_equally_shaped(self.v_to_average):
            done = len(self.v_to_average) >= self.spin_averaging.value()
            __v = np.mean(self.v_to_average, axis=0) if len(self.v_to_average) > 1 else self.v_to_average[0]
            if t.shape[0] and __v.shape[0]:
                if t.shape[0] > __v.shape[0]:
                    t = t[: __v.shape[0]]
                self.line_dc.setData(t, __v)  # - offsets[self.combo_data_channel.currentText()],
                self.line_ac.setData(t, __v - np.nanmedian(__v))
                if __v.shape[0] > 6:
                    v_1: F64Array = __v[: __v.shape[0] // 3]
                    v_2: F64Array = __v[__v.shape[0] // 3 :]
                    noise: np.float64 = np.nanstd(v_1, dtype=np.float64)
                    self.label_stats_noise.setValue(noise)
                    self.label_stats_magnitude.setValue(np.ptp(v_2) - 2.0 * noise)
                    self.label_stats_area.setValue((np.nanmean(v_2) - np.nanmean(v_1)) * (t[-1] - t[0]))
                else:
                    self.label_stats_noise.clear()
                    self.label_stats_magnitude.clear()
                    self.label_stats_area.clear()

        return done


if __name__ == "__main__":
    app: QApplication = QApplication(sys.argv)
    if QT5:
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    window: App = App()
    window.show()
    app.exec()
