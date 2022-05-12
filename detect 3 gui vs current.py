# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from app_base.detect import DetectBase
from backend.utils import zero_sources


class App(DetectBase):
    def setup_ui_appearance(self) -> None:
        super(App, self).setup_ui_appearance()

        self.figure.getAxis('bottom').setLabel(text='Current', units='nA')

    @property
    def stat_file(self) -> Path:
        return self.saving_location / (' '.join((
            'detect',
            self.config.get('output', 'prefix', fallback=''),
            f'{self.temperature * 1e3:.6f}'.rstrip('0').rstrip('.') + 'mK',
            f'{self.power_dbm:.6f}'.rstrip('0').rstrip('.') + 'dBm',
            f'CC{self.cycles_count}',
            f'{self.frequency:.6f}'.rstrip('0').rstrip('.') + 'GHz'
            if self.synthesizer.output else '',
            f'P{self.pulse_duration:.6f}'.rstrip('0').rstrip('.') + 's',
            f'WaP{self.waiting_after_pulse:.6f}'.rstrip('0').rstrip('.') + 's',
            f'ST{self.setting_time:.6f}'.rstrip('0').rstrip('.') + 's',
            self.config.get('output', 'suffix', fallback='')
        )).replace('  ', ' ').strip(' ') + '.txt')

    @property
    def _line_index(self) -> int:
        return (self.power_index
                + (self.frequency_index * len(self.frequency_values)
                   + self.temperature_index) * len(self.temperature_values))

    @property
    def _line_name(self) -> str:
        return ', '.join((
            f'{self.power_dbm:.6f}'.rstrip('0').rstrip('.') + 'dBm'
            if not np.isnan(self.power_dbm) else '',
            f'{self.frequency:.6f}'.rstrip('0').rstrip('.') + 'GHz'
            if not np.isnan(self.frequency) else '',
            f'{self.temperature * 1e3:.6f}'.rstrip('0').rstrip('.') + 'mK',
        )).replace('  ', ' ').replace('  ', ' ').strip(', ')

    def _next_indices(self) -> bool:
        if self.stop_key_power.isChecked():
            self.on_button_stop_clicked()
            return False
        self.power_index += 1
        while self.check_exists and self._stat_file_exists():
            self.power_index += 1
        if self.power_index >= len(self.power_dbm_values):
            self.power_index = 0
            if self.stop_key_frequency.isChecked():
                self.on_button_stop_clicked()
                return False
            self.frequency_index += 1
            while self.check_exists and self._stat_file_exists():
                self.frequency_index += 1
            if self.frequency_index >= len(self.frequency_values):
                self.frequency_index = 0
                if self.stop_key_temperature.isChecked():
                    self.on_button_stop_clicked()
                    return False
                self.temperature_index += 1
                while self.check_exists and self._stat_file_exists():
                    self.temperature_index += 1
                if self.temperature_index >= len(self.temperature_values):
                    self.temperature_index = 0
                    self.on_button_stop_clicked()
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

        prob: float = np.inf
        err: float
        while not self.results_queue.empty():
            prob, err = self.results_queue.get(block=True)
            self._add_plot_point(self.bias_current, prob, err)

        self._watch_temperature()

        self.good_to_measure.buf[0] &= not self.button_pause.isChecked()

        if not self.measurement.is_alive():
            self.timer.stop()
            if self.stop_key_bias.isChecked():
                self.on_button_stop_clicked()
                return
            self.bias_current_index += 1
            if self.bias_current_index >= len(self.bias_current_values):
                self.bias_current_index = 0
                if prob < self.minimal_probability_to_measure:
                    current_bias: float = self.bias_current
                    while self.bias_current <= current_bias:
                        if not self._next_indices():
                            self.on_button_stop_clicked()
                            return
                elif not self._next_indices():
                    self.on_button_stop_clicked()
                    return
            self.triton.issue_temperature(6, self.temperature)
            self.label_temperature.setValue(self.temperature * 1000)
            self.synthesizer.frequency = self.frequency * 1e9
            self.label_frequency.setValue(self.frequency)
            self.synthesizer.power.level = self.power_dbm
            self.label_power.setValue(self.power_dbm)
            self.label_bias.setValue(self.bias_current)

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
