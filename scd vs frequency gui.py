# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from app_base.scd import SwitchingCurrentDistributionBase
from backend.utils import zero_sources


class App(SwitchingCurrentDistributionBase):
    def setup_ui_appearance(self) -> None:
        super(App, self).setup_ui_appearance()

        self.canvas_mean.getAxis('bottom').setLabel(text='Frequency', units='GHz')
        self.canvas_std.getAxis('bottom').setLabel(text='Frequency', units='GHz')

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
            if self.synthesizer_output and len(self.frequency_values) == 1 else '',
            f'{self.power_dbm:.6f}'.rstrip('0').rstrip('.') + 'dBm'
            if self.synthesizer_output else '',
            f'from {self.initial_biases[-1]:.6f}'.rstrip('0').rstrip('.') + 'nA',
            f'threshold{self.trigger_voltage * 1e3:.8f}'.rstrip('0').rstrip('.') + 'mV',
            self.config.get('output', 'suffix', fallback=''),
        )).replace('  ', ' ').replace('  ', ' ').strip(' ') + '.txt')

    @property
    def _line_index(self) -> int:
        return self.power_index + self.temperature_index * len(self.power_dbm_values)

    @property
    def _line_name(self) -> str:
        return ', '.join((
            f'{self.temperature * 1e3:.6f}'.rstrip('0').rstrip('.') + 'mK',
            f'{self.power_dbm:.6f}'.rstrip('0').rstrip('.') + 'dBm'
            if not np.isnan(self.power_dbm) else '',
        )).replace('  ', ' ').replace('  ', ' ').strip(' ').rstrip(',')

    def _next_indices(self) -> bool:
        if self.stop_key_frequency.isChecked():
            return False
        self.frequency_index += 1
        while self.synthesizer_output and self.check_exists and self._data_file_exists():
            self.frequency_index += 1
        if not self.synthesizer_output or self.frequency_index >= len(self.frequency_values):
            self.frequency_index = 0
            if self.stop_key_power.isChecked():
                return False
            self.power_index += 1
            while self.synthesizer_output and self.check_exists and self._data_file_exists():
                self.power_index += 1
            if not self.synthesizer_output or self.power_index >= len(self.power_dbm_values):
                self.power_index = 0
                if self.stop_key_temperature.isChecked():
                    return False
                self.temperature_index += 1
                while self.check_exists and self._data_file_exists():
                    self.temperature_index += 1
                if self.temperature_index >= len(self.temperature_values):
                    self.temperature_index = 0
                    return False
                actual_temperature: float
                temperature_unit: str
                actual_temperature, temperature_unit = self.triton.query_temperature(6)
                if not ((1.0 - 0.01 * self.temperature_tolerance) * self.temperature
                        < actual_temperature
                        < (1.0 + 0.01 * self.temperature_tolerance) * self.temperature):
                    self.temperature_just_set = True
        return True

    def on_timeout(self) -> None:
        self._read_state_queue()
        self._read_switching_data_queue()

        mean: float
        std: float
        while not self.results_queue.empty():
            mean, std = self.results_queue.get(block=True)
            self._add_plot_point(self.frequency, mean, std)

        self._watch_temperature()

        self.good_to_measure.buf[0] &= not self.button_pause.isChecked()

        if not self.measurement.is_alive():
            self.timer.stop()
            if not self._next_indices():
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
