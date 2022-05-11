# -*- coding: utf-8 -*-
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from numpy.typing import NDArray

from app_base.scd import SwitchingCurrentDistributionBase
from backend.utils import error, zero_sources


class App(SwitchingCurrentDistributionBase):
    def setup_ui_appearance(self) -> None:
        super(App, self).setup_ui_appearance()

        self.canvas_mean.getAxis('bottom').setLabel(text='Power', units='dBm')
        self.canvas_std.getAxis('bottom').setLabel(text='Power', units='dBm')

    @property
    def stat_file(self) -> Path:
        return self.saving_location / (' '.join((
            'SCD-stat',
            self.config.get('output', 'prefix', fallback=''),
            f'{self.temperature * 1e3:.6f}'.rstrip('0').rstrip('.') + 'mK',
            f'v{self.current_speed:.6f}'.rstrip('0').rstrip('.') + 'nAps',
            f'd{self.delay_between_cycles:.6f}'.rstrip('0').rstrip('.') + 's',
            f'CC{self.cycles_count}',
            f'{self.frequency:.6f}'.rstrip('0').rstrip('.') + 'GHz'
            if self.synthesizer_output else '',
            f'{self.power_dbm:.6f}'.rstrip('0').rstrip('.') + 'dBm'
            if self.synthesizer_output and len(self.power_dbm_values) == 1 else '',
            f'from {self.initial_biases[-1]:.6f}'.rstrip('0').rstrip('.') + 'nA',
            f'threshold{self.trigger_voltage * 1e3:.8f}'.rstrip('0').rstrip('.') + 'mV',
            self.config.get('output', 'suffix', fallback=''),
        )).replace('  ', ' ').replace('  ', ' ').strip(' ') + '.txt')

    @property
    def _line_index(self) -> int:
        return self.frequency_index + self.temperature_index * len(self.frequency_values)

    @property
    def _line_name(self) -> str:
        return ', '.join((
            f'{self.temperature * 1e3:.6f}'.rstrip('0').rstrip('.') + 'mK',
            f'{self.frequency:.6f}'.rstrip('0').rstrip('.') + 'GHz'
            if not np.isnan(self.frequency) else '',
        )).replace('  ', ' ').replace('  ', ' ').strip(' ').rstrip(',')

    def _next_indices(self) -> bool:
        self.frequency_index += 1
        if self.frequency_index >= len(self.frequency_values):
            self.frequency_index = 0
            self.power_index += 1
            if self.power_index >= len(self.power_dbm_values):
                self.power_index = 0
                self.temperature_index += 1
                if self.temperature_index >= len(self.temperature_values):
                    self.temperature_index = 0
                    return False
        return True

    def on_timeout(self) -> None:
        cycle_index: int
        remaining_time: timedelta
        while not self.state_queue.empty():
            cycle_index, remaining_time = self.state_queue.get(block=True)
            self.label_loop_number.setValue(cycle_index)
            self.label_remaining_time.setText(str(remaining_time)[:9])

        current: np.float64
        voltage: np.float64
        while not self.switching_data_queue.empty():
            current, voltage = self.switching_data_queue.get(block=True)
            self.switching_current.append(current)
            self.switching_voltage.append(voltage)
            self.label_mean_current.setValue(np.nanmean(self.switching_current) * 1e9)
            self.label_std_current.setValue(np.nanstd(self.switching_current) * 1e9)

        mean: float
        std: float
        while not self.results_queue.empty():
            old_x_data: NDArray[np.float64] = (np.empty(0)
                                               if self.plot_line_mean.xData is None
                                               else self.plot_line_mean.xData)
            old_mean_data: NDArray[np.float64] = (np.empty(0)
                                                  if self.plot_line_mean.yData is None
                                                  else self.plot_line_mean.yData)
            old_std_data: NDArray[np.float64] = (np.empty(0)
                                                 if self.plot_line_std.yData is None
                                                 else self.plot_line_std.yData)
            mean, std = self.results_queue.get(block=True)
            x_data: NDArray[np.float64] = np.concatenate((old_x_data, [self.power_dbm]))
            mean_data: NDArray[np.float64] = np.concatenate((old_mean_data, [mean]))
            std_data: NDArray[np.float64] = np.concatenate((old_std_data, [std]))
            self.plot_line_mean.setData(x_data, mean_data)
            self.plot_line_std.setData(x_data, std_data)

        actual_temperature: float
        temperature_unit: str
        actual_temperature, temperature_unit = self.triton.query_temperature(6)
        if not ((1.0 - 0.01 * self.temperature_tolerance) * self.temperature
                < actual_temperature
                < (1.0 + 0.01 * self.temperature_tolerance) * self.temperature):
            self.good_to_measure.buf[0] = False
            self.bad_temperature_time = datetime.now()
            self.timer.setInterval(1000)
            print(f'temperature {actual_temperature} {temperature_unit} '
                  f'is too far from {self.temperature:.3f} K')
            if not self.triton.issue_temperature(6, self.temperature):
                error(f'failed to set temperature to {self.temperature} K')
                self.timer.stop()
                self.measurement.terminate()
            if self.change_filtered_readings:
                if not self.triton.issue_filter_readings(6, self.triton.filter_readings(self.temperature)):
                    error(f'failed to change the state of filtered readings')
                    self.timer.stop()
                    self.measurement.terminate()
            if not self.triton.issue_heater_range(6, self.triton.heater_range(self.temperature)):
                error(f'failed to change the heater range')
                self.timer.stop()
                self.measurement.terminate()
        elif self.temperature_just_set:
            td: timedelta = datetime.now() - self.bad_temperature_time
            if td > self.temperature_delay:
                self.timer.setInterval(50)
                self.good_to_measure.buf[0] = True
                self.temperature_just_set = False
            else:
                self.good_to_measure.buf[0] = False
                print(f'temperature {actual_temperature} {temperature_unit} '
                      f'is close enough to {self.temperature:.3f} K, but not for long enough yet'
                      f': {self.temperature_delay - td} left')
                self.timer.setInterval(1000)
        else:
            self.good_to_measure.buf[0] = True

        self.good_to_measure.buf[0] &= not self.button_pause.isChecked()

        if not self.measurement.is_alive():
            self.timer.stop()
            if self.stop_key_frequency.isChecked():
                self.on_button_stop_clicked()
                return
            if self.synthesizer_output and self.check_exists:
                while self.frequency_index < len(self.frequency_values) and self.data_file.exists():
                    self.frequency_index += 1
            else:
                self.frequency_index += 1
            if not self.synthesizer_output or self.frequency_index >= len(self.frequency_values):
                self.frequency_index = 0
                if self.stop_key_power.isChecked():
                    self.on_button_stop_clicked()
                    return
                if self.synthesizer_output and self.check_exists:
                    while self.power_index < len(self.power_dbm_values) and self.data_file.exists():
                        self.power_index += 1
                else:
                    self.power_index += 1
                if not self.synthesizer_output or self.power_index >= len(self.power_dbm_values):
                    self.power_index = 0
                    if self.stop_key_temperature.isChecked():
                        self.on_button_stop_clicked()
                        return
                    if self.check_exists:
                        while self.temperature_index < len(self.temperature_values) and self.data_file.exists():
                            self.temperature_index += 1
                    else:
                        self.temperature_index += 1
                    if self.temperature_index >= len(self.temperature_values):
                        self.temperature_index = 0
                        self.on_button_stop_clicked()
                        return
                    self.triton.issue_temperature(6, self.temperature)
                    self.label_temperature.setValue(self.temperature * 1000)
                self.synthesizer.frequency = self.frequency * 1e9
                self.label_frequency.setValue(self.frequency)
            self.synthesizer.power.level = self.power_dbm
            self.label_power.setValue(self.power_dbm)

            self.start_measurement()
        else:
            self.timer.setInterval(50)


if __name__ == '__main__':
    app: QApplication = QApplication(sys.argv)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    window: App = App()
    window.show()
    app.exec()
    zero_sources()
