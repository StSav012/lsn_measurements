# -*- coding: utf-8 -*-
from __future__ import annotations

import pyqtgraph as pg
from qtpy.QtCore import QSettings, Qt
from qtpy.QtGui import QCloseEvent, QIcon
from qtpy.QtWidgets import (QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout,
                            QWidget)

from ui.histogram import Histogram
from ui.safe_button import SafeButton

__all__ = ['SwitchingCurrentDistributionGUI']


class SwitchingCurrentDistributionGUI(QMainWindow):
    def __init__(self, parent: QWidget | None = None, flags: Qt.WindowFlags = Qt.WindowFlags()) -> None:
        super().__init__(parent=parent, flags=flags)

        self.settings: QSettings = QSettings("SavSoft", "Switching Current Distribution", self)

        self.setWindowTitle(self.tr('Switching Current Distribution'))
        self.setWindowIcon(QIcon('lsn.svg'))

        self.central_widget: QWidget = QWidget(self)
        self.main_layout: QHBoxLayout = QHBoxLayout(self.central_widget)
        self.controls_layout: QVBoxLayout = QVBoxLayout()
        self.parameters_box: QGroupBox = QGroupBox(self.central_widget)
        self.parameters_layout: QFormLayout = QFormLayout(self.parameters_box)
        self.button_topmost: QPushButton = QPushButton(self.central_widget)
        self.button_drop_measurement: SafeButton = SafeButton('', timeout=4.0, parent=self.central_widget)
        self.histogram: Histogram = Histogram(self.central_widget)
        self.stop_sings_box: QGroupBox = QGroupBox(self.central_widget)
        self.stop_sings_box.setLayout(QVBoxLayout())
        self.buttons_layout: QHBoxLayout = QHBoxLayout()

        self.figure: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget(self.central_widget)
        self.canvas_mean: pg.PlotItem = self.figure.addPlot(row=0, col=0)
        self.canvas_std: pg.PlotItem = self.figure.addPlot(row=1, col=0)
        self.plot_lines_mean: dict[int, pg.PlotDataItem] = dict()
        self.plot_lines_std: dict[int, pg.PlotDataItem] = dict()

        self.label_loop_number: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_remaining_time: QLabel = QLabel(self.central_widget)
        self.label_mean_current: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_std_current: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_power: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_frequency: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_current_speed: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_delay_between_cycles: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_temperature: pg.ValueLabel = pg.ValueLabel(self.central_widget)

        self.stop_key_power: QPushButton = QPushButton(self.stop_sings_box)
        self.stop_key_frequency: QPushButton = QPushButton(self.stop_sings_box)
        self.stop_key_current_speed: QPushButton = QPushButton(self.stop_sings_box)
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

        x_axis = self.canvas_mean.getAxis('bottom')
        x_axis.enableAutoSIPrefix(False)
        y_axis = self.canvas_mean.getAxis('left')
        y_axis.setLabel(text=self.tr('Mean'), units=self.tr('nA'))
        y_axis.enableAutoSIPrefix(False)
        self.canvas_mean.ctrl.averageGroup.setChecked(False)
        self.canvas_mean.showGrid(x=True, y=True)

        x_axis = self.canvas_std.getAxis('bottom')
        x_axis.enableAutoSIPrefix(False)
        y_axis = self.canvas_std.getAxis('left')
        y_axis.setLabel(text=self.tr('StD'), units=self.tr('nA'))
        y_axis.enableAutoSIPrefix(False)
        self.canvas_std.ctrl.averageGroup.setChecked(False)
        self.canvas_std.showGrid(x=True, y=True)
        self.canvas_std.setXLink(self.canvas_mean)

        self.canvas_mean.addLegend()
        self.canvas_std.addLegend()

        self.label_loop_number.formatStr = '{value:.0f}'
        self.label_mean_current.suffix = self.tr('nA')
        self.label_mean_current.formatStr = '{value:0.2f} {suffix}'
        self.label_std_current.suffix = self.tr('nA')
        self.label_std_current.formatStr = '{value:0.2f} {suffix}'
        self.label_power.suffix = self.tr('dBm')
        self.label_frequency.suffix = self.tr('GHz')
        self.label_frequency.formatStr = '{value:0.4f} {suffix}'
        self.label_current_speed.suffix = self.tr('ms')
        self.label_current_speed.formatStr = '{value:0.1f} {suffix}'
        self.label_delay_between_cycles.suffix = self.tr('ms')
        self.label_delay_between_cycles.formatStr = '{value:0.1f} {suffix}'
        self.label_temperature.suffix = self.tr('mK')
        self.label_temperature.formatStr = '{value:0.2f} {suffix}'

        self.figure.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self.main_layout.addWidget(self.figure)
        self.main_layout.addLayout(self.controls_layout)
        self.main_layout.setStretch(0, 1)
        self.controls_layout.addWidget(self.parameters_box)
        self.controls_layout.addWidget(self.button_topmost)
        self.controls_layout.addWidget(self.button_drop_measurement)
        self.controls_layout.addWidget(self.histogram)
        self.controls_layout.addWidget(self.stop_sings_box)
        self.controls_layout.addLayout(self.buttons_layout)

        self.parameters_layout.addRow(self.tr('Remaining time:'), self.label_remaining_time)
        self.parameters_layout.addRow(self.tr('Mean current:'), self.label_mean_current)
        self.parameters_layout.addRow(self.tr('Current std:'), self.label_std_current)
        self.parameters_layout.addRow(self.tr('Frequency:'), self.label_frequency)
        self.parameters_layout.addRow(self.tr('Power:'), self.label_power)
        self.parameters_layout.addRow(self.tr('Current speed:'), self.label_current_speed)
        self.parameters_layout.addRow(self.tr('Delay b/w cycles:'), self.label_delay_between_cycles)
        self.parameters_layout.addRow(self.tr('Temperature:'), self.label_temperature)
        self.parameters_layout.addRow(self.tr('Loop number:'), self.label_loop_number)

        self.button_topmost.setText(self.tr('Keep the Window Topmost'))
        self.button_topmost.setCheckable(True)

        self.button_drop_measurement.setText(self.tr('Next Measurement'))

        self.histogram.set_label(text=self.tr('Current'), unit=self.tr('A'))

        self.stop_key_power.setText(self.tr('Stop after this Power'))
        self.stop_key_frequency.setText(self.tr('Stop after this Frequency'))
        self.stop_key_current_speed.setText(self.tr('Stop after this Current Speed'))
        self.stop_key_delay_between_cycles.setText(self.tr('Stop after this Delay'))
        self.stop_key_temperature.setText(self.tr('Stop after this Temperature'))

        self.stop_key_power.setCheckable(True)
        self.stop_key_frequency.setCheckable(True)
        self.stop_key_current_speed.setCheckable(True)
        self.stop_key_delay_between_cycles.setCheckable(True)
        self.stop_key_temperature.setCheckable(True)

        self.stop_sings_box.layout().addWidget(self.stop_key_power)
        self.stop_sings_box.layout().addWidget(self.stop_key_frequency)
        self.stop_sings_box.layout().addWidget(self.stop_key_current_speed)
        self.stop_sings_box.layout().addWidget(self.stop_key_delay_between_cycles)
        self.stop_sings_box.layout().addWidget(self.stop_key_temperature)

        self.buttons_layout.addWidget(self.button_start)
        self.buttons_layout.addWidget(self.button_pause)
        self.buttons_layout.addWidget(self.button_stop)

        self.button_start.setText(self.tr('Start'))
        self.button_pause.setText(self.tr('Pause'))
        self.button_pause.setCheckable(True)
        self.button_stop.setText(self.tr('Stop'))
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
