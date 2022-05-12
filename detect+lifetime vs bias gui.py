# -*- coding: utf-8 -*-
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication)
from numpy.typing import NDArray

from app_base.detect_lifetime import DetectLifetimeBase
from backend.utils import error, zero_sources


class App(DetectLifetimeBase):
    def setup_ui_appearance(self) -> None:
        super(App, self).setup_ui_appearance()

        self.canvas_detect.getAxis('bottom').setLabel(text='Power', units='dBm')
        self.canvas_lifetime.getAxis('bottom').setLabel(text='Current', units='nA')

    @property
    def stat_file_detect(self) -> Path:
        return self.saving_location / (' '.join((
            'detect',
            self.config.get('output', 'prefix', fallback=''),
            f'{self.temperature * 1e3:.6f}'.rstrip('0').rstrip('.') + 'mK',
            f'{self.bias_current:.6f}'.rstrip('0').rstrip('.') + 'nA',
            f'CC{self.cycles_count_detect}',
            f'{self.frequency:.6f}'.rstrip('0').rstrip('.') + 'GHz'
            if self.synthesizer.output else '',
            f'P{self.pulse_duration:.6f}'.rstrip('0').rstrip('.') + 's',
            f'WaP{self.waiting_after_pulse:.6f}'.rstrip('0').rstrip('.') + 's',
            f'ST{self.setting_time:.6f}'.rstrip('0').rstrip('.') + 's',
            self.config.get('output', 'suffix', fallback='')
        )).replace('  ', ' ').strip(' ') + '.txt')

    @property
    def stat_file_lifetime(self) -> Path:
        return self.saving_location / (' '.join((
            'lifetime',
            self.config.get('output', 'prefix', fallback=''),
            f'{self.temperature * 1e3:.6f}'.rstrip('0').rstrip('.') + 'mK',
            f'{self.bias_current_values[0]:.6f}'.rstrip('0').rstrip('.') + 'nA'
            if len(self.bias_current_values) == 1 else '',
            f'd{self.delay_between_cycles:.6f}'.rstrip('0').rstrip('.') + 's',
            f'CC{self.cycles_count_lifetime}',
            f'ST{self.setting_time:.6f}'.rstrip('0').rstrip('.') + 's',
            f'{self.frequency:.6f}'.rstrip('0').rstrip('.') + 'GHz'
            if not np.isnan(self.frequency) else '',
            f'{self.power_dbm:.6f}'.rstrip('0').rstrip('.') + 'dBm'
            if not np.isnan(self.power_dbm) else '',
            f'from {self.initial_biases[-1]:.6f}'.rstrip('0').rstrip('.') + 'nA',
            self.config.get('output', 'suffix', fallback=''),
        )).replace('  ', ' ').replace('  ', ' ').strip(' ') + '.txt')

    @property
    def stat_file(self) -> Path:
        return {'detect': self.stat_file_detect, 'lifetime': self.stat_file_lifetime}[self.mode]

    @property
    def _line_index_detect(self) -> int:
        return (self.bias_current_index
                + (self.frequency_index * len(self.frequency_values)
                   + self.temperature_index) * len(self.temperature_values))

    @property
    def _line_name_detect(self) -> str:
        return ', '.join((
            f'{self.bias_current:.6f}'.rstrip('0').rstrip('.') + 'nA',
            f'{self.temperature * 1e3:.6f}'.rstrip('0').rstrip('.') + 'mK',
            f'{self.frequency:.6f}'.rstrip('0').rstrip('.') + 'GHz'
            if not np.isnan(self.frequency) else '',
        )).replace('  ', ' ').replace('  ', ' ').strip(', ')

    @property
    def _line_index_lifetime(self) -> int:
        return self.temperature_index

    @property
    def _line_name_lifetime(self) -> str:
        return ', '.join((
            f'{self.temperature * 1e3:.6f}'.rstrip('0').rstrip('.') + 'mK',
        )).replace('  ', ' ').replace('  ', ' ').strip(', ')

    def _next_indices(self) -> bool:
        self.power_index += 1
        if self.mode == 'lifetime' or self.power_index >= len(self.power_dbm_values):
            self.power_index = 0
            self.frequency_index += 1
            if self.mode == 'lifetime' or self.frequency_index >= len(self.frequency_values):
                self.frequency_index = 0
                self.temperature_index += 1
                if self.temperature_index >= len(self.temperature_values):
                    self.temperature_index = 0
                    return False
        return True

    def on_timeout(self) -> None:
        cycle_index: int
        estimated_cycles_count: int
        switches_count: int
        while not self.state_queue_detect.empty():
            cycle_index, estimated_cycles_count, switches_count = self.state_queue_detect.get(block=True)
            self.label_loop_number.setValue(cycle_index + 1)
            self.label_loop_count.setValue(estimated_cycles_count)
            self.label_probability.setValue(switches_count / (cycle_index + 1) * 100)

        prob: float = 146.
        err: float
        while not self.results_queue_detect.empty():
            old_x_data: NDArray[np.float64] = (np.empty(0, dtype=np.float64)
                                               if self.plot_line_detect.xData is None
                                               else self.plot_line_detect.xData)
            old_y_data: NDArray[np.float64] = (np.empty(0, dtype=np.float64)
                                               if self.plot_line_detect.yData is None
                                               else self.plot_line_detect.yData)
            prob, err = self.results_queue_detect.get(block=True)
            x_data: NDArray[np.float64] = np.concatenate((old_x_data, [self.power_dbm]))
            y_data: NDArray[np.float64] = np.concatenate((old_y_data, [prob]))
            self.plot_line_detect.setData(x_data, y_data)

        cycle_index: int
        spent_time: timedelta
        while not self.state_queue_lifetime.empty():
            cycle_index, spent_time = self.state_queue_lifetime.get(block=True)
            self.label_loop_number.setValue(cycle_index + 1)
            self.label_spent_time.setValue(spent_time.total_seconds())

        set_bias: float
        lifetime_0: float
        lifetime: float
        while not self.results_queue_lifetime.empty():
            old_x_data: NDArray[np.float64] = (np.empty(0, dtype=np.float64)
                                               if self.plot_line_lifetime.xData is None
                                               else self.plot_line_lifetime.xData)
            old_y_data: NDArray[np.float64] = (np.empty(0, dtype=np.float64)
                                               if self.plot_line_lifetime.yData is None
                                               else self.plot_line_lifetime.yData)
            set_bias, lifetime_0, lifetime = self.results_queue_lifetime.get(block=True)
            x_data: NDArray[np.float64] = np.append(old_x_data, set_bias)
            y_data: NDArray[np.float64] = np.append(old_y_data, lifetime)
            self.plot_line_lifetime.setData(x_data, y_data)
            self.last_lifetime_0 = lifetime_0

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

            if self.mode == 'detect' and self.power_index == len(self.power_dbm_values) - 1:
                sys.stderr.write('switching to lifetime\n'.upper())
                self.mode = 'lifetime'
                self.start_measurement()
                return
            elif self.mode == 'lifetime':
                sys.stderr.write('switching to detect\n'.upper())
                self.mode = 'detect'

            if self.stop_key_power.isChecked():
                self.on_button_stop_clicked()
                return
            self.power_index += 1
            if prob < self.minimal_probability_to_measure or self.power_index == len(self.power_dbm_values):
                self.power_index = 0
                if self.stop_key_bias.isChecked():
                    self.on_button_stop_clicked()
                    return
                if self.check_exists:
                    while self.bias_current_index < len(self.bias_current_values) and self.stat_file.exists():
                        self.bias_current_index += 1
                else:
                    self.bias_current_index += 1
                if self.bias_current_index >= len(self.bias_current_values):
                    self.bias_current_index = 0
                    if self.stop_key_frequency.isChecked():
                        self.on_button_stop_clicked()
                        return
                    if self.check_exists:
                        while self.frequency_index < len(self.frequency_values) and self.stat_file.exists():
                            self.frequency_index += 1
                    else:
                        self.frequency_index += 1
                    if self.frequency_index >= len(self.frequency_values):
                        self.frequency_index = 0
                        if self.stop_key_temperature.isChecked():
                            self.on_button_stop_clicked()
                            return
                        if self.check_exists:
                            while self.temperature_index < len(self.temperature_values) and self.stat_file.exists():
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
                self.label_bias.setValue(self.bias_current)
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
