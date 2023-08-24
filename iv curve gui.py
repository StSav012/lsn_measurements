# coding: utf-8
from __future__ import annotations

import sys
from multiprocessing import Queue

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from qtpy.QtCore import QSettings, QTimer, Qt
from qtpy.QtGui import QCloseEvent, QIcon
from qtpy.QtWidgets import (QApplication, QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QMainWindow, QPushButton,
                            QRadioButton, QStatusBar, QVBoxLayout, QWidget)

from backend.hardware import device_adc
from backend.measurement.iv_curve import IVCurveMeasurement


class GUI(QMainWindow):
    def __init__(self, flags=Qt.WindowFlags()) -> None:
        super(GUI, self).__init__(flags=flags)

        self.settings: QSettings = QSettings("SavSoft", "IV Curve", self)

        self.setWindowTitle('IV Curve')
        self.setWindowIcon(QIcon('lsn.svg'))

        self.central_widget: QWidget = QWidget(self, flags=Qt.WindowFlags())
        self.main_layout: QHBoxLayout = QHBoxLayout(self.central_widget)
        self.controls_layout: QVBoxLayout = QVBoxLayout()
        self.parameters_box: QGroupBox = QGroupBox(self.central_widget)
        self.parameters_layout: QFormLayout = QFormLayout(self.parameters_box)
        self.buttons_layout: QHBoxLayout = QHBoxLayout()

        self.figure: pg.PlotWidget = pg.PlotWidget(self.central_widget)
        self.plot_line: pg.PlotDataItem = self.figure.plot(np.empty(0))

        self.spin_current_min: pg.SpinBox = pg.SpinBox(self.central_widget)
        self.spin_current_max: pg.SpinBox = pg.SpinBox(self.central_widget)
        self.spin_current_rate: pg.SpinBox = pg.SpinBox(self.central_widget)
        self.spin_current_divider: pg.SpinBox = pg.SpinBox(self.central_widget)
        self.check_two_way: QCheckBox = QCheckBox(self.central_widget)
        self.spin_ballast_resistance: pg.SpinBox = pg.SpinBox(self.central_widget)
        self.spin_resistance_in_series: pg.SpinBox = pg.SpinBox(self.central_widget)
        self.spin_voltage_gain: pg.SpinBox = pg.SpinBox(self.central_widget)
        self.spin_adc_rate: pg.SpinBox = pg.SpinBox(self.central_widget)

        self.frame_current_mode: QGroupBox = QGroupBox(self.central_widget)
        self.radio_linear_mode: QRadioButton = QRadioButton(self.central_widget)
        self.radio_parabolic_mode: QRadioButton = QRadioButton(self.central_widget)

        self.button_start: QPushButton = QPushButton(self.central_widget)
        self.button_stop: QPushButton = QPushButton(self.central_widget)
        self.button_filter: QPushButton = QPushButton(self.central_widget)

        self.status_bar: QStatusBar = QStatusBar(self.central_widget)
        self.setStatusBar(self.status_bar)

        self.setup_ui_appearance()
        self.load_settings()
        self.setup_actions()

    def setup_ui_appearance(self) -> None:
        opts: dict[str, bool | str | int]
        opts = {
            'suffix': 'A',
            'siPrefix': True,
            'decimals': 3,
            'dec': True,
            'compactHeight': False,
            'format': '{scaledValue:.{decimals}f}{suffixGap}{siPrefix}{suffix}'
        }
        self.spin_current_min.setOpts(**opts)
        self.spin_current_max.setOpts(**opts)
        opts = {
            'suffix': 'A/s',
            'siPrefix': True,
            'decimals': 3,
            'dec': True,
            'compactHeight': False,
            'format': '{scaledValue:.{decimals}f}{suffixGap}{siPrefix}{suffix}'
        }
        self.spin_current_rate.setOpts(**opts)
        opts = {
            'suffix': 'Ω',
            'siPrefix': True,
            'decimals': 6,
            'dec': True,
            'compactHeight': False,
            'format': '{scaledValue:.{decimals}f}{suffixGap}{siPrefix}{suffix}'
        }
        self.spin_ballast_resistance.setOpts(**opts)
        self.spin_resistance_in_series.setOpts(**opts)

        opts = {
            'suffix': '',
            'siPrefix': False,
            'decimals': 3,
            'dec': True,
            'compactHeight': False,
            'format': '{value:.{decimals}f}'
        }
        self.spin_current_divider.setOpts(**opts)
        self.spin_voltage_gain.setOpts(**opts)

        opts = {
            'bounds': (device_adc.ai_min_rate, device_adc.ai_max_multi_chan_rate),
            'suffix': 'Hz',
            'siPrefix': True,
            'decimals': 3,
            'dec': True,
            'compactHeight': False,
            'format': '{scaledValue:.{decimals}f}{suffixGap}{siPrefix}{suffix}'
        }
        self.spin_adc_rate.setOpts(**opts)

        self.figure.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.figure.getAxis('bottom').setLabel(text='Voltage', units='V')
        self.figure.getAxis('left').setLabel(text='Current', units='A')
        self.figure.plotItem.ctrl.xGridCheck.setChecked(True)
        self.figure.plotItem.ctrl.yGridCheck.setChecked(True)

        self.main_layout.addWidget(self.figure)
        self.main_layout.addLayout(self.controls_layout)
        self.main_layout.setStretch(0, 1)
        self.controls_layout.addWidget(self.parameters_box)
        self.controls_layout.addLayout(self.buttons_layout)

        self.parameters_layout.addRow('Minimum current:', self.spin_current_min)
        self.parameters_layout.addRow('Maximum current:', self.spin_current_max)
        self.parameters_layout.addRow('Current rate:', self.spin_current_rate)
        self.parameters_layout.addRow('Current divider:', self.spin_current_divider)
        self.check_two_way.setText('Two-way')
        self.parameters_layout.addWidget(self.check_two_way)
        self.parameters_layout.addRow('Ballast resistance:', self.spin_ballast_resistance)
        self.parameters_layout.addRow('Resistance in series:', self.spin_resistance_in_series)
        self.parameters_layout.addRow('Voltage gain:', self.spin_voltage_gain)
        self.parameters_layout.addRow('ADC rate:', self.spin_adc_rate)
        self.parameters_layout.addRow('Current mode:', self.frame_current_mode)

        self.frame_current_mode.setLayout(QVBoxLayout())
        self.frame_current_mode.layout().addWidget(self.radio_linear_mode)
        self.frame_current_mode.layout().addWidget(self.radio_parabolic_mode)
        self.radio_linear_mode.setText('Linear')
        self.radio_parabolic_mode.setText('Parabolic')

        self.buttons_layout.addWidget(self.button_start)
        self.buttons_layout.addWidget(self.button_stop)
        self.buttons_layout.addWidget(self.button_filter)

        self.button_start.setText('Start')
        self.button_stop.setText('Stop')
        self.button_filter.setText('Filter')
        self.button_stop.setDisabled(True)
        self.button_filter.setDisabled(True)

        self.setCentralWidget(self.central_widget)

    def setup_actions(self):
        self.button_start.clicked.connect(self.on_button_start_clicked)
        self.button_stop.clicked.connect(self.on_button_stop_clicked)
        self.button_filter.clicked.connect(self.on_button_filter_clicked)

    def load_settings(self) -> None:
        self.restoreGeometry(self.settings.value('windowGeometry', b''))
        self.restoreState(self.settings.value('windowState', b''))

        self.settings.beginGroup('parameters')
        self.spin_current_min.setValue(self.settings.value('minCurrent', 0.0, float))
        self.spin_current_max.setValue(self.settings.value('maxCurrent', 0.0, float))
        self.spin_current_rate.setValue(self.settings.value('currentRate', 1e-8, float))
        self.spin_current_divider.setValue(self.settings.value('currentDivider', 1, float))
        self.check_two_way.setChecked(self.settings.value('twoWay', True, bool))
        self.spin_ballast_resistance.setValue(self.settings.value('ballastResistance', 2e6, float))
        self.spin_resistance_in_series.setValue(self.settings.value('resistanceInSeries', 0.0, float))
        self.spin_voltage_gain.setValue(self.settings.value('voltageGain', 100, float))
        self.spin_adc_rate.setValue(self.settings.value('ADCRate', 50, float))
        self.radio_parabolic_mode.setChecked(self.settings.value('currentMode', 'parabolic', str) == 'parabolic')
        self.radio_linear_mode.setChecked(self.settings.value('currentMode', 'linear', str) == 'linear')
        self.settings.endGroup()

    def save_settings(self) -> None:
        self.settings.setValue('windowGeometry', self.saveGeometry())
        self.settings.setValue('windowState', self.saveState())

        self.settings.beginGroup('parameters')
        self.settings.setValue('minCurrent', self.spin_current_min.value())
        self.settings.setValue('maxCurrent', self.spin_current_max.value())
        self.settings.setValue('currentRate', self.spin_current_rate.value())
        self.settings.setValue('currentDivider', self.spin_current_divider.value())
        self.settings.setValue('twoWay', self.check_two_way.isChecked())
        self.settings.setValue('ballastResistance', self.spin_ballast_resistance.value())
        self.settings.setValue('resistanceInSeries', self.spin_resistance_in_series.value())
        self.settings.setValue('voltageGain', self.spin_voltage_gain.value())
        self.settings.setValue('ADCRate', self.spin_adc_rate.value())
        self.settings.setValue('currentMode', 'linear' if self.radio_linear_mode.isChecked() else 'parabolic')
        self.settings.endGroup()

        self.settings.sync()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.save_settings()
        event.accept()

    def on_button_start_clicked(self) -> None:
        self.button_start.setDisabled(True)
        self.parameters_box.setDisabled(True)
        self.button_stop.setEnabled(True)
        self.button_filter.setDisabled(True)

    def on_button_stop_clicked(self) -> None:
        self.button_filter.setEnabled(True)
        self.button_stop.setDisabled(True)
        self.parameters_box.setEnabled(True)
        self.button_start.setEnabled(True)

    def on_button_filter_clicked(self) -> None:
        pass


class App(GUI):
    def __init__(self, flags=Qt.WindowFlags()) -> None:
        super(App, self).__init__(flags=flags)

        self.timer: QTimer = QTimer(self)
        self.timer.timeout.connect(self.on_timeout)

        self.results_queue: Queue[NDArray[np.float64]] = Queue()
        self.measurement: IVCurveMeasurement | None = None

    def on_button_start_clicked(self) -> None:
        super(App, self).on_button_start_clicked()
        self.plot_line.clear()
        self.timer.start(10)
        self.measurement = IVCurveMeasurement(
                self.results_queue,
                min_current=self.spin_current_min.value(),
                max_current=self.spin_current_max.value(),
                current_rate=self.spin_current_rate.value(),
                two_way=self.check_two_way.isChecked(),
                ballast_resistance=self.spin_ballast_resistance.value(),
                voltage_gain=self.spin_voltage_gain.value(),
                adc_rate=self.spin_adc_rate.value(),
                current_divider=self.spin_current_divider.value(),
                resistance_in_series=self.spin_resistance_in_series.value(),
                current_mode='linear' if self.radio_linear_mode.isChecked() else 'parabolic'
        )
        self.measurement.start()

    def on_button_stop_clicked(self) -> None:
        self.measurement.terminate()
        self.measurement.join()
        self.timer.stop()
        if self.plot_line.xData is not None and self.plot_line.yData is not None:
            x_data: NDArray[np.float64] = self.plot_line.xData
            y_data: NDArray[np.float64] = self.plot_line.yData
            k, b = list(np.polyfit(y_data, x_data, 1))
            self.status_bar.showMessage(f'Average resistance is {k} Ω')
        super(App, self).on_button_stop_clicked()

    def on_button_filter_clicked(self) -> None:
        if self.plot_line.xData is None or self.plot_line.yData is None:
            return
        import pandas as pd
        x_data: NDArray[np.float64] = np.empty(0) if self.plot_line.xData is None else self.plot_line.xData
        y_data: NDArray[np.float64] = np.empty(0) if self.plot_line.yData is None else self.plot_line.yData
        median: int = round(2e6 / 102400)
        data: pd.DataFrame = pd.DataFrame({'x': x_data, 'y': y_data})
        rolling_median: pd.DataFrame = \
            (pd.DataFrame(data)
             .rolling(window=median, center=True, axis=0)
             .median()[median // 2:-(median // 2)]
             .drop_duplicates(ignore_index=True))
        x_data = rolling_median['x']
        y_data = rolling_median['y']
        self.plot_line.setData(x_data, y_data)

        super(App, self).on_button_stop_clicked()

    def on_timeout(self) -> None:
        while not self.results_queue.empty():
            old_x_data: NDArray[np.float64] = np.empty(0) if self.plot_line.xData is None else self.plot_line.xData
            old_y_data: NDArray[np.float64] = np.empty(0) if self.plot_line.yData is None else self.plot_line.yData
            new_data: NDArray[np.float64] = self.results_queue.get()
            x_data: NDArray[np.float64] = np.concatenate((old_x_data, new_data[1]))
            y_data: NDArray[np.float64] = np.concatenate((old_y_data, new_data[0]))
            self.plot_line.setData(x_data, y_data)
        if not self.measurement.is_alive():
            self.on_button_stop_clicked()


if __name__ == '__main__':
    app: QApplication = QApplication(sys.argv)
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    window: App = App()
    window.show()
    app.exec()
