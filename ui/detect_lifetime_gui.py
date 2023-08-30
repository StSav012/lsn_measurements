# -*- coding: utf-8 -*-
from __future__ import annotations

import pyqtgraph as pg
from qtpy.QtCore import QSettings, Qt
from qtpy.QtGui import QCloseEvent, QIcon
from qtpy.QtWidgets import QFormLayout, QGroupBox, QHBoxLayout, QMainWindow, QPushButton, QVBoxLayout, QWidget

from ui.safe_button import SafeButton

__all__ = ['DetectLifetimeGUI']


class DetectLifetimeGUI(QMainWindow):
    def __init__(self, parent: QWidget | None = None, flags: Qt.WindowFlags = Qt.WindowFlags()) -> None:
        super().__init__(parent=parent, flags=flags)

        self.settings: QSettings = QSettings("SavSoft", "Detect+Lifetime", self)

        self.setWindowTitle('Detect+Lifetime')
        self.setWindowIcon(QIcon('lsn.svg'))

        self.central_widget: QWidget = QWidget(self)
        self.main_layout: QHBoxLayout = QHBoxLayout(self.central_widget)
        self.controls_layout: QVBoxLayout = QVBoxLayout()
        self.parameters_box: QGroupBox = QGroupBox(self.central_widget)
        self.parameters_layout: QFormLayout = QFormLayout(self.parameters_box)
        self.button_topmost: QPushButton = QPushButton(self.central_widget)
        self.button_drop_measurement: SafeButton = SafeButton('', timeout=4.0, parent=self.central_widget)
        self.stop_sings_box: QGroupBox = QGroupBox(self.central_widget)
        self.stop_sings_box.setLayout(QVBoxLayout())
        self.buttons_layout: QHBoxLayout = QHBoxLayout()

        self.figure: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget(self.central_widget)
        self.canvas_detect: pg.PlotItem = self.figure.addPlot(row=0, col=0)
        self.canvas_lifetime: pg.PlotItem = self.figure.addPlot(row=0, col=1)
        self.plot_lines_detect: dict[int, pg.PlotDataItem] = dict()
        self.plot_lines_lifetime: dict[int, pg.PlotDataItem] = dict()

        self.label_loop_number: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_loop_count: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_probability: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_spent_time: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_bias: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_power: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_frequency: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_pulse_duration: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_setting_time: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_delay_between_cycles: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_temperature: pg.ValueLabel = pg.ValueLabel(self.central_widget)

        self.stop_key_bias: QPushButton = QPushButton(self.stop_sings_box)
        self.stop_key_power: QPushButton = QPushButton(self.stop_sings_box)
        self.stop_key_frequency: QPushButton = QPushButton(self.stop_sings_box)
        self.stop_key_setting_time: QPushButton = QPushButton(self.stop_sings_box)
        self.stop_key_delay_between_cycles: QPushButton = QPushButton(self.stop_sings_box)
        self.stop_key_temperature: QPushButton = QPushButton(self.stop_sings_box)

        self.button_start: QPushButton = QPushButton(self.central_widget)
        self.button_pause: QPushButton = QPushButton(self.central_widget)
        self.button_stop: QPushButton = QPushButton(self.central_widget)

        self.setup_ui_appearance()
        self.load_settings()
        self.setup_actions()

    def setup_ui_appearance(self) -> None:
        x_axis: pg.AxisItem
        y_axis: pg.AxisItem

        x_axis = self.canvas_detect.getAxis('bottom')
        x_axis.enableAutoSIPrefix(False)
        y_axis = self.canvas_detect.getAxis('left')
        y_axis.setLabel(text='Probability', units='%')
        y_axis.enableAutoSIPrefix(False)
        self.canvas_detect.ctrl.averageGroup.setChecked(False)
        self.canvas_detect.ctrl.logYCheck.setChecked(True)
        self.canvas_detect.ctrl.xGridCheck.setChecked(True)
        self.canvas_detect.ctrl.yGridCheck.setChecked(True)

        x_axis = self.canvas_lifetime.getAxis('bottom')
        x_axis.enableAutoSIPrefix(False)
        y_axis = self.canvas_lifetime.getAxis('left')
        y_axis.setLabel(text='Lifetime', units='s')
        y_axis.enableAutoSIPrefix(False)
        self.canvas_lifetime.ctrl.averageGroup.setChecked(False)
        self.canvas_lifetime.ctrl.logYCheck.setChecked(True)
        self.canvas_lifetime.ctrl.xGridCheck.setChecked(True)
        self.canvas_lifetime.ctrl.yGridCheck.setChecked(True)

        self.canvas_detect.addLegend(offset=(-30, -30))
        self.canvas_lifetime.addLegend(offset=(30, -30))

        self.label_spent_time.suffix = 's'
        self.label_spent_time.formatStr = '{value:0.5f} {suffix}'
        self.label_loop_count.formatStr = '{value:.0f}'
        self.label_loop_number.formatStr = '{value:.0f}'
        self.label_probability.suffix = '%'
        self.label_probability.formatStr = '{value:0.2f}{suffix}'
        self.label_power.suffix = 'dBm'
        self.label_frequency.suffix = 'GHz'
        self.label_frequency.formatStr = '{value:0.4f} {suffix}'
        self.label_pulse_duration.suffix = 'ms'
        self.label_pulse_duration.formatStr = '{value:0.4f} {suffix}'
        self.label_bias.suffix = 'nA'
        self.label_bias.formatStr = '{value:0.2f} {suffix}'
        self.label_setting_time.suffix = 'ms'
        self.label_setting_time.formatStr = '{value:0.1f} {suffix}'
        self.label_delay_between_cycles.suffix = 'ms'
        self.label_delay_between_cycles.formatStr = '{value:0.1f} {suffix}'
        self.label_temperature.suffix = 'mK'
        self.label_temperature.formatStr = '{value:0.2f} {suffix}'

        self.main_layout.addWidget(self.figure)
        self.main_layout.addLayout(self.controls_layout)
        self.main_layout.setStretch(0, 1)
        self.controls_layout.addWidget(self.parameters_box)
        self.controls_layout.addWidget(self.button_topmost)
        self.controls_layout.addWidget(self.button_drop_measurement)
        self.controls_layout.addStretch(1)
        self.controls_layout.addWidget(self.stop_sings_box)
        self.controls_layout.addLayout(self.buttons_layout)

        self.parameters_layout.addRow('Since bias set:', self.label_spent_time)
        self.parameters_layout.addRow('Bias current:', self.label_bias)
        self.parameters_layout.addRow('Frequency:', self.label_frequency)
        self.parameters_layout.addRow('Power:', self.label_power)
        self.parameters_layout.addRow('Pulse duration:', self.label_pulse_duration)
        self.parameters_layout.addRow('Setting time:', self.label_setting_time)
        self.parameters_layout.addRow('Delay b/w cycles:', self.label_delay_between_cycles)
        self.parameters_layout.addRow('Temperature:', self.label_temperature)
        self.parameters_layout.addRow('Loop count:', self.label_loop_count)
        self.parameters_layout.addRow('Loop number:', self.label_loop_number)
        self.parameters_layout.addRow('Probability:', self.label_probability)

        self.button_topmost.setText('Keep the Window Topmost')
        self.button_topmost.setCheckable(True)

        self.button_drop_measurement.setText('Next Measurement')

        self.stop_key_bias.setText('Stop after this Bias')
        self.stop_key_power.setText('Stop after this Power')
        self.stop_key_frequency.setText('Stop after this Frequency')
        self.stop_key_setting_time.setText('Stop after this Setting Time')
        self.stop_key_delay_between_cycles.setText('Stop after this Delay')
        self.stop_key_temperature.setText('Stop after this Temperature')

        self.stop_key_bias.setCheckable(True)
        self.stop_key_power.setCheckable(True)
        self.stop_key_frequency.setCheckable(True)
        self.stop_key_setting_time.setCheckable(True)
        self.stop_key_delay_between_cycles.setCheckable(True)
        self.stop_key_temperature.setCheckable(True)

        self.stop_sings_box.layout().addWidget(self.stop_key_bias)
        self.stop_sings_box.layout().addWidget(self.stop_key_power)
        self.stop_sings_box.layout().addWidget(self.stop_key_frequency)
        self.stop_sings_box.layout().addWidget(self.stop_key_setting_time)
        self.stop_sings_box.layout().addWidget(self.stop_key_delay_between_cycles)
        self.stop_sings_box.layout().addWidget(self.stop_key_temperature)

        self.buttons_layout.addWidget(self.button_start)
        self.buttons_layout.addWidget(self.button_pause)
        self.buttons_layout.addWidget(self.button_stop)

        self.button_start.setText('Start')
        self.button_pause.setText('Pause')
        self.button_pause.setCheckable(True)
        self.button_stop.setText('Stop')
        self.button_stop.setDisabled(True)

        self.setCentralWidget(self.central_widget)

    def setup_actions(self):
        self.button_topmost.toggled.connect(self.on_button_topmost_toggled)
        self.button_start.clicked.connect(self.on_button_start_clicked)
        self.button_stop.clicked.connect(self.on_button_stop_clicked)

    def load_settings(self) -> None:
        self.restoreGeometry(self.settings.value('windowGeometry', b''))
        self.restoreState(self.settings.value('windowState', b''))

    def save_settings(self) -> None:
        self.settings.setValue('windowGeometry', self.saveGeometry())
        self.settings.setValue('windowState', self.saveState())
        self.settings.sync()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.on_button_stop_clicked()
        self.save_settings()
        event.accept()

    def on_button_topmost_toggled(self, on: bool) -> None:
        if on:
            self.setWindowFlags(self.windowFlags()
                                | Qt.WindowFlags.CustomizeWindowHint | Qt.WindowFlags.WindowStaysOnTopHint)
            self.show()
        else:
            self.setWindowFlags(self.windowFlags()
                                ^ (Qt.WindowFlags.CustomizeWindowHint | Qt.WindowFlags.WindowStaysOnTopHint))
            self.show()

    def on_button_start_clicked(self) -> None:
        self.button_start.setDisabled(True)
        self.button_pause.setChecked(False)
        self.button_stop.setEnabled(True)

    def on_button_stop_clicked(self) -> None:
        self.button_stop.setDisabled(True)
        self.button_start.setEnabled(True)
