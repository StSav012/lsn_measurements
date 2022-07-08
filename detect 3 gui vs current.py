# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from numpy.typing import NDArray

from app_base.detect import DetectBase
from backend.utils import load_txt, zero_sources
from backend.utils.string_utils import format_float


class App(DetectBase):
    def setup_ui_appearance(self) -> None:
        super(App, self).setup_ui_appearance()

        self.figure.getAxis('bottom').setLabel(text='Current', units='nA')

    @property
    def stat_file(self) -> Path:
        return self.saving_location / (' '.join(filter(None, (
            'detect',
            self.config.get('output', 'prefix', fallback=''),
            format_float(self.temperature * 1e3, suffix='mK'),
            format_float(self.power_dbm, suffix='dBm'),
            f'CC{self.cycles_count}',
            format_float(self.frequency, suffix='GHz'),
            format_float(self.pulse_duration, prefix='P', suffix='s'),
            format_float(self.waiting_after_pulse, prefix='WaP', suffix='s'),
            format_float(self.setting_time, prefix='ST', suffix='s'),
            self.config.get('output', 'suffix', fallback='')
        ))) + '.txt')

    @property
    def _line_index(self) -> int:
        return (self.power_index
                + (self.frequency_index * len(self.frequency_values)
                   + self.temperature_index) * len(self.temperature_values))

    @property
    def _line_name(self) -> str:
        return ', '.join(filter(None, (
            format_float(self.power_dbm, suffix='dBm'),
            format_float(self.frequency, suffix='GHz'),
            format_float(self.temperature * 1e3, suffix='mK'),
        )))

    def _fill_the_data_from_stat_file(self) -> None:
        data: NDArray[float]
        titles: tuple[str]
        data, titles = load_txt(self.stat_file, sep='\t', encoding='utf-8')
        if 'Set Bias Current [nA]' in titles and 'Switch Probability [%]' in titles \
                and 'Probability Uncertainty [%]' in titles:
            x_column: int = titles.index('Set Bias Current [nA]')
            prob_column: int = titles.index('Switch Probability [%]')
            err_column: int = titles.index('Probability Uncertainty [%]')
            for x, prob, err in zip(data[..., x_column], data[..., prob_column], data[..., err_column]):
                self._add_plot_point(x, prob, err)

    def _next_indices(self, make_step: bool = True) -> bool:
        if self.stop_key_power.isChecked():
            return False
        if make_step:
            self.power_index += 1
        while self.check_exists and self._stat_file_exists():
            self.power_index += 1
        if self.power_index >= len(self.power_dbm_values):
            self.power_index = 0
            if self.stop_key_frequency.isChecked():
                return False
            if make_step:
                self.frequency_index += 1
            while self.check_exists and self._stat_file_exists():
                self.frequency_index += 1
            if self.frequency_index >= len(self.frequency_values):
                self.frequency_index = 0
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
