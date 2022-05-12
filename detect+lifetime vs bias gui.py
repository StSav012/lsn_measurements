# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from app_base.detect_lifetime import DetectLifetimeBase
from backend.utils import zero_sources


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

    def _next_indices(self, make_step: bool = True) -> bool:
        if self.stop_key_bias.isChecked():
            return False
        if make_step:
            self.bias_current_index += 1
        while self.check_exists and self._stat_file_exists():
            self.bias_current_index += 1
        if self.bias_current_index >= len(self.bias_current_values):
            self.bias_current_index = 0
            if self.stop_key_frequency.isChecked():
                return False
            self.frequency_index += 1
            while self.check_exists and self._stat_file_exists():
                self.frequency_index += 1
            if self.frequency_index >= len(self.frequency_values):
                self.frequency_index = 0
                if self.stop_key_temperature.isChecked():
                    return False
                self.temperature_index += 1
                while self.check_exists and self._stat_file_exists():
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
        self._read_state_queue_detect()

        prob: float = 146.
        err: float
        while not self.results_queue_detect.empty():
            prob, err = self.results_queue_detect.get(block=True)
            self._add_plot_point_detect(self.power_dbm, prob, err)

        self._read_state_queue_lifetime()

        set_bias: float
        lifetime_0: float
        lifetime: float
        while not self.results_queue_lifetime.empty():
            set_bias, lifetime_0, lifetime = self.results_queue_lifetime.get(block=True)
            self._add_plot_point_lifetime(set_bias, lifetime)
            self.last_lifetime_0 = lifetime_0

        self._watch_temperature()

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
            if prob < self.minimal_probability_to_measure or self.power_index >= len(self.power_dbm_values):
                self.power_index = 0
                if not self._next_indices():
                    self.on_button_stop_clicked()
                    return

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
