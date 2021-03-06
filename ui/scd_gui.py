# -*- coding: utf-8 -*-
from typing import Dict

import pyqtgraph as pg
from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtGui import QCloseEvent, QIcon
from PyQt5.QtWidgets import (QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout,
                             QWidget)

__all__ = ['SwitchingCurrentDistributionGUI']


class SwitchingCurrentDistributionGUI(QMainWindow):
    def __init__(self, flags=Qt.WindowFlags()) -> None:
        super().__init__(flags=flags)

        self.settings: QSettings = QSettings("SavSoft", "Switching Current Distribution", self)

        self.setWindowTitle('Switching Current Distribution')
        self.setWindowIcon(QIcon('lsn.svg'))

        self.central_widget: QWidget = QWidget(self)
        self.main_layout: QHBoxLayout = QHBoxLayout(self.central_widget)
        self.controls_layout: QVBoxLayout = QVBoxLayout()
        self.parameters_box: QGroupBox = QGroupBox(self.central_widget)
        self.parameters_layout: QFormLayout = QFormLayout(self.parameters_box)
        self.button_topmost: QPushButton = QPushButton(self.central_widget)
        self.stop_sings_box: QGroupBox = QGroupBox(self.central_widget)
        self.stop_sings_box.setLayout(QVBoxLayout())
        self.buttons_layout: QHBoxLayout = QHBoxLayout()

        self.figure: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget(self.central_widget)
        self.canvas_mean: pg.PlotItem = self.figure.addPlot(row=0, col=0)
        self.canvas_std: pg.PlotItem = self.figure.addPlot(row=1, col=0)
        self.plot_lines_mean: Dict[int, pg.PlotDataItem] = dict()
        self.plot_lines_std: Dict[int, pg.PlotDataItem] = dict()

        self.label_loop_number: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_remaining_time: QLabel = QLabel(self.central_widget)
        self.label_mean_current: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_std_current: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_power: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_frequency: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_temperature: pg.ValueLabel = pg.ValueLabel(self.central_widget)

        self.stop_key_power: QPushButton = QPushButton(self.stop_sings_box)
        self.stop_key_frequency: QPushButton = QPushButton(self.stop_sings_box)
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
        y_axis.setLabel(text='Mean', units='nA')
        y_axis.enableAutoSIPrefix(False)
        self.canvas_mean.ctrl.averageGroup.setChecked(False)
        self.canvas_mean.ctrl.xGridCheck.setChecked(True)
        self.canvas_mean.ctrl.yGridCheck.setChecked(True)

        x_axis = self.canvas_std.getAxis('bottom')
        x_axis.enableAutoSIPrefix(False)
        y_axis = self.canvas_std.getAxis('left')
        y_axis.setLabel(text='StD', units='nA')
        y_axis.enableAutoSIPrefix(False)
        self.canvas_std.ctrl.averageGroup.setChecked(False)
        self.canvas_std.ctrl.xGridCheck.setChecked(True)
        self.canvas_std.ctrl.yGridCheck.setChecked(True)
        self.canvas_std.setXLink(self.canvas_mean)

        self.canvas_mean.addLegend()
        self.canvas_std.addLegend()

        self.label_loop_number.formatStr = '{value:.0f}'
        self.label_mean_current.suffix = 'nA'
        self.label_mean_current.formatStr = '{value:0.2f} {suffix}'
        self.label_std_current.suffix = 'nA'
        self.label_std_current.formatStr = '{value:0.2f} {suffix}'
        self.label_power.suffix = 'dBm'
        self.label_frequency.suffix = 'GHz'
        self.label_frequency.formatStr = '{value:0.4f} {suffix}'
        self.label_temperature.suffix = 'mK'
        self.label_temperature.formatStr = '{value:0.2f} {suffix}'

        self.figure.setFocusPolicy(Qt.ClickFocus)

        self.main_layout.addWidget(self.figure)
        self.main_layout.addLayout(self.controls_layout)
        self.main_layout.setStretch(0, 1)
        self.controls_layout.addWidget(self.parameters_box)
        self.controls_layout.addWidget(self.button_topmost)
        self.controls_layout.addStretch(1)
        self.controls_layout.addWidget(self.stop_sings_box)
        self.controls_layout.addLayout(self.buttons_layout)

        self.parameters_layout.addRow('Remaining time:', self.label_remaining_time)
        self.parameters_layout.addRow('Mean current:', self.label_mean_current)
        self.parameters_layout.addRow('Current std:', self.label_std_current)
        self.parameters_layout.addRow('Frequency:', self.label_frequency)
        self.parameters_layout.addRow('Power:', self.label_power)
        self.parameters_layout.addRow('Temperature:', self.label_temperature)
        self.parameters_layout.addRow('Loop number:', self.label_loop_number)

        self.button_topmost.setText('Keep the Window Topmost')
        self.button_topmost.setCheckable(True)

        self.stop_key_power.setText('Stop after this Power')
        self.stop_key_frequency.setText('Stop after this Frequency')
        self.stop_key_temperature.setText('Stop after this Temperature')

        self.stop_key_power.setCheckable(True)
        self.stop_key_frequency.setCheckable(True)
        self.stop_key_temperature.setCheckable(True)

        self.stop_sings_box.layout().addWidget(self.stop_key_power)
        self.stop_sings_box.layout().addWidget(self.stop_key_frequency)
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
            self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint)
            self.show()
        else:
            self.setWindowFlags(self.windowFlags() ^ (Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint))
            self.show()

    def on_button_start_clicked(self) -> None:
        self.button_start.setDisabled(True)
        self.button_pause.setChecked(False)
        self.button_stop.setEnabled(True)

    def on_button_stop_clicked(self) -> None:
        self.button_stop.setDisabled(True)
        self.button_start.setEnabled(True)
