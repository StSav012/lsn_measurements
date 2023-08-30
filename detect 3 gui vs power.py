# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from app_base.detect import DetectBase
from backend.utils import zero_sources
from backend.utils.string_utils import format_float


class App(DetectBase):
    def setup_ui_appearance(self) -> None:
        super(App, self).setup_ui_appearance()

        self.figure.getAxis('bottom').setLabel(text=self.tr('Power'), units=self.tr('dBm'))

    @property
    def stat_file(self) -> Path:
        return self.saving_location / (' '.join(filter(None, (
            'detect',
            self.config.get('output', 'prefix', fallback=''),
            format_float(self.temperature * 1e3, suffix='mK'),
            format_float(self.bias_current, suffix='nA'),
            f'CC{self.cycles_count}',
            format_float(self.frequency, suffix='GHz'),
            format_float(self.pulse_duration, prefix='P', suffix='s'),
            format_float(self.waiting_after_pulse, prefix='WaP', suffix='s'),
            format_float(self.setting_time, prefix='ST', suffix='s'),
            self.config.get('output', 'suffix', fallback='')
        ))) + '.txt')

    @property
    def _line_index(self) -> int:
        return (self.bias_current_index
                + (self.setting_time_index
                   + (self.frequency_index
                      * len(self.bias_current_values)
                      + self.temperature_index
                      ) * len(self.frequency_values)
                   ) * len(self.setting_time_values)
                )

    @property
    def _line_name(self) -> str:
        return ', '.join(filter(None, (
            format_float(self.bias_current, suffix=self.tr('nA')),
            format_float(self.frequency, suffix=self.tr('GHz')),
            format_float(self.setting_time * 1e3, prefix='ST ', suffix=self.tr('ms')),
            format_float(self.temperature * 1e3, suffix=self.tr('mK')),
        )))

    def _add_plot_point_from_file(self) -> None:
        if self.data_file in self.saved_files:
            return
        self.saved_files.add(self.data_file)
        measured_data: NDArray[float] = self._get_data_file_content()
        bias_current: NDArray[float] = measured_data[0]
        median_bias_current: float = cast(float, np.nanmedian(bias_current))
        min_reasonable_bias_current: float = median_bias_current * (1. - .01 * self.max_reasonable_bias_error)
        max_reasonable_bias_current: float = median_bias_current * (1. + .01 * self.max_reasonable_bias_error)
        reasonable: NDArray[np.bool_] = ((bias_current >= min_reasonable_bias_current)
                                         & (bias_current <= max_reasonable_bias_current))
        good_count: int = np.count_nonzero(reasonable)
        prob: float = 100.0 * good_count / self.cycles_count
        err: float = np.sqrt(prob * (100.0 - prob) / self.cycles_count)
        self._add_plot_point(cast(float, np.mean(bias_current)), prob, err)

    def _next_indices(self, make_step: bool = True) -> bool:
        if self.stop_key_bias.isChecked():
            return False
        if make_step:
            self.bias_current_index += 1
        while self.check_exists and self._data_file_exists():
            self._add_plot_point_from_file()
            self.bias_current_index += 1
        if self.bias_current_index >= len(self.bias_current_values):
            self.bias_current_index = 0
            if self.stop_key_frequency.isChecked():
                return False
            if make_step:
                self.frequency_index += 1
            while self.check_exists and self._data_file_exists():
                self._add_plot_point_from_file()
                self.frequency_index += 1
            if self.frequency_index >= len(self.frequency_values):
                self.frequency_index = 0
                if self.stop_key_setting_time.isChecked():
                    return False
                if make_step:
                    self.setting_time_index += 1
                while self.check_exists and self._data_file_exists():
                    self._add_plot_point_from_file()
                    self.setting_time_index += 1
                if self.setting_time_index >= len(self.setting_time_values):
                    self.setting_time_index = 0
                    if self.stop_key_temperature.isChecked():
                        return False
                    if make_step:
                        self.temperature_index += 1
                    while self.check_exists and self._data_file_exists():
                        self._add_plot_point_from_file()
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
            self._add_plot_point(self.power_dbm, prob, err)

        self._watch_temperature()

        self.good_to_measure.buf[0] &= not self.button_pause.isChecked()
        self.good_to_measure.buf[127] = self.button_drop_measurement.isPushed()

        if not self.measurement.is_alive():
            self.button_drop_measurement.reset()
            self.timer.stop()
            if self.stop_key_power.isChecked():
                self.on_button_stop_clicked()
                return
            self.power_index += 1
            if self.power_index >= len(self.power_dbm_values):
                self.power_index = 0
                if prob < self.minimal_probability_to_measure:
                    current_power: float = self.power_dbm
                    while self.power_dbm <= current_power:
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
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    window: App = App()
    window.show()
    app.exec()
    zero_sources()
