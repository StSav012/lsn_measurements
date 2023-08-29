# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from app_base.detect_lifetime import DetectLifetimeBase
from backend.utils import zero_sources
from backend.utils.string_utils import format_float


class App(DetectLifetimeBase):
    def setup_ui_appearance(self) -> None:
        super(App, self).setup_ui_appearance()

        self.canvas_detect.getAxis('bottom').setLabel(text='Power', units='dBm')
        self.canvas_lifetime.getAxis('bottom').setLabel(text='Current', units='nA')

    @property
    def stat_file_detect(self) -> Path:
        return self.saving_location / (' '.join(filter(None, (
            'detect',
            self.config.get('output', 'prefix', fallback=''),
            format_float(self.temperature * 1e3, suffix='mK'),
            format_float(self.bias_current, suffix='nA'),
            f'CC{self.cycles_count_detect}',
            format_float(self.frequency, suffix='GHz'),
            format_float(self.pulse_duration, prefix='P', suffix='s'),
            format_float(self.waiting_after_pulse, prefix='WaP', suffix='s'),
            format_float(self.setting_time, prefix='ST', suffix='s'),
            self.config.get('output', 'suffix', fallback='')
        ))) + '.txt')

    @property
    def stat_file_lifetime(self) -> Path:
        return self.saving_location / (' '.join(filter(None, (
            'lifetime',
            self.config.get('output', 'prefix', fallback=''),
            format_float(self.temperature * 1e3, suffix='mK'),
            format_float(self.bias_current_values[0], suffix='nA')
            if len(self.bias_current_values) == 1 else '',
            format_float(self.delay_between_cycles, prefix='d', suffix='s'),
            f'CC{self.cycles_count_lifetime}',
            format_float(self.setting_time, prefix='ST', suffix='s'),
            format_float(self.frequency, suffix='GHz')
            if not np.isnan(self.frequency) else '',
            format_float(self.power_dbm, suffix='dBm')
            if not np.isnan(self.power_dbm) else '',
            format_float(self.initial_biases[-1], prefix='from ', suffix='nA'),
            self.config.get('output', 'suffix', fallback=''),
        ))) + '.txt')

    @property
    def stat_file(self) -> Path:
        return {'detect': self.stat_file_detect, 'lifetime': self.stat_file_lifetime}[self.mode]

    @property
    def _line_index_detect(self) -> int:
        return (self.bias_current_index
                + (self.temperature_index
                   + (self.frequency_index
                      ) * len(self.frequency_values)
                   ) * len(self.temperature_values)
                )

    @property
    def _line_name_detect(self) -> str:
        return ', '.join(filter(None, (
            format_float(self.bias_current, suffix='nA'),
            format_float(self.temperature * 1e3, suffix='mK'),
            format_float(self.delay_between_cycles * 1e3, suffix='ms'),
            format_float(self.frequency, suffix='GHz')
            if not np.isnan(self.frequency) else '',
        )))

    @property
    def _line_index_lifetime(self) -> int:
        return (self.temperature_index
                + (self.delay_between_cycles_index
                   ) * len(self.delay_between_cycles_values)
                )

    @property
    def _line_name_lifetime(self) -> str:
        return ', '.join(filter(None, (
            f'{self.temperature * 1e3:.6f}'.rstrip('0').rstrip('.') + 'mK',
            format_float(self.delay_between_cycles * 1e3, suffix="ms"),
        )))

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
            if make_step:
                self.frequency_index += 1
            while self.check_exists and self._stat_file_exists():
                self.frequency_index += 1
            if self.frequency_index >= len(self.frequency_values):
                self.frequency_index = 0
                if self.stop_key_delay_between_cycles.isChecked():
                    return False
                if make_step:
                    self.delay_between_cycles_index += 1
                while self.check_exists and self._stat_file_exists():
                    self.delay_between_cycles_index += 1
                if self.delay_between_cycles_index >= len(self.delay_between_cycles_values):
                    self.delay_between_cycles_index = 0
                    if self.stop_key_temperature.isChecked():
                        return False
                    if make_step:
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
        self.good_to_measure.buf[127] = self.button_drop_measurement.isPushed()

        if not self.measurement.is_alive():
            self.button_drop_measurement.reset()
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
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    window: App = App()
    window.show()
    app.exec()
    zero_sources()
