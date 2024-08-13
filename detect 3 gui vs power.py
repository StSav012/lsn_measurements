# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
from typing import final

import numpy as np
from astropy.units import K, Quantity
from numpy.typing import NDArray
from pyqtgraph.functions import intColor
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QApplication

from app_base.detect import DetectBase
from utils.ni import zero_sources
from utils.string_utils import format_float


@final
class App(DetectBase):
    def setup_ui_appearance(self) -> None:
        super(App, self).setup_ui_appearance()

        self.figure.getAxis("bottom").setLabel(text=self.tr("Power"), units=self.tr("dBm"))

    @property
    def stat_file(self) -> Path:
        return self.saving_location / (
            " ".join(
                filter(
                    None,
                    (
                        "detect",
                        self.config.get("output", "prefix", fallback=""),
                        format_float(self.temperature * 1e3, suffix="mK"),
                        format_float(self.bias_current, suffix="nA"),
                        format_float(self.frequency, suffix="GHz"),
                        format_float(self.power_dbm, suffix="dBm") if len(self.power_dbm_values) == 1 else "",
                        f"CC{self.cycles_count}",
                        format_float(self.pulse_duration, prefix="P", suffix="s"),
                        format_float(self.waiting_after_pulse, prefix="WaP", suffix="s"),
                        format_float(self.setting_time, prefix="ST", suffix="s"),
                        self.config.get("output", "suffix", fallback=""),
                    ),
                )
            )
            + ".txt"
        )

    # fmt: off
    @property
    def _line_index(self) -> int:
        return (self.bias_current_index
                + (self.frequency_index
                   + (self.pulse_duration_index
                      + (self.setting_time_index
                         + (self.temperature_index
                            ) * (len(self.setting_time_values) or 1)
                         ) * (len(self.pulse_duration_values) or 1)
                      ) * (len(self.frequency_values) or 1)
                   ) * (len(self.bias_current_values) or 1)
                )
    # fmt: on

    @property
    def _line_name(self) -> str:
        return ", ".join(
            filter(
                None,
                (
                    format_float(self.bias_current, suffix=self.tr("nA")) if len(self.bias_current_values) > 1 else "",
                    format_float(self.frequency, suffix=self.tr("GHz")) if len(self.frequency_values) > 1 else "",
                    format_float(self.power_dbm, suffix=self.tr("dBm")) if len(self.power_dbm_values) == 1 else "",
                    (
                        format_float(self.pulse_duration * 1e3, prefix=self.tr("P "), suffix=self.tr("ms"))
                        if len(self.pulse_duration_values) > 1
                        else ""
                    ),
                    (
                        format_float(self.setting_time * 1e3, prefix=self.tr("ST "), suffix=self.tr("ms"))
                        if len(self.setting_time_values) > 1
                        else ""
                    ),
                    (
                        format_float(self.temperature * 1e3, suffix=self.tr("mK"))
                        if len(self.temperature_values) > 1
                        else ""
                    ),
                ),
            )
        )

    def _line_color(self, index: int) -> QColor:
        hues: int = len(self.bias_current_values) or 1
        if hues < 7:
            hues *= len(self.frequency_values) or 1
        if hues < 7:
            hues *= len(self.pulse_duration_values) or 1
        if hues < 7:
            hues *= len(self.setting_time_values) or 1
        if hues < 7:
            hues *= len(self.temperature_values) or 1
        return intColor(index, hues=hues)

    def _add_plot_point_from_file(self) -> None:
        if self.data_file in self.saved_files:
            return
        self.saved_files.add(self.data_file)
        measured_data: NDArray[float] = self._get_data_file_content()
        if measured_data.shape[0] == 4 and measured_data.shape[1]:
            switches_count: int = measured_data.shape[1]
            actual_cycles_count: int = measured_data[0, -1].item()
            prob: float = 100.0 * switches_count / actual_cycles_count if actual_cycles_count > 0 else np.nan
            err: float = np.sqrt(prob * (100.0 - prob) / actual_cycles_count) if actual_cycles_count > 0 else np.nan
            self._add_plot_point(self.power_dbm, prob, err)

    def _next_indices(self) -> bool:
        while True:
            if self.stop_key_power.isChecked():
                return False
            while self.check_exists and self._data_file_exists():
                self._add_plot_point_from_file()
                self.power_index += 1
            if (
                np.isnan(self.last_prob) or self.last_prob > self.minimal_probability_to_measure
            ) and self.power_index < len(self.power_dbm_values):
                return True
            self.power_index = 0

            if self.stop_key_bias.isChecked():
                return False
            self.bias_current_index += 1
            if self.bias_current_index < len(self.bias_current_values):
                continue
            self.bias_current_index = 0

            if self.stop_key_frequency.isChecked():
                return False
            self.frequency_index += 1
            if self.frequency_index < len(self.frequency_values):
                continue
            self.frequency_index = 0

            if self.stop_key_pulse_duration.isChecked():
                return False
            self.pulse_duration_index += 1
            if self.pulse_duration_index < len(self.pulse_duration_values):
                continue
            self.pulse_duration_index = 0

            if self.stop_key_setting_time.isChecked():
                return False
            self.setting_time_index += 1
            if self.setting_time_index < len(self.setting_time_values):
                continue
            self.setting_time_index = 0

            if self.stop_key_temperature.isChecked():
                return False
            try:
                self.temperature_index += 1
                if self.temperature_index < len(self.temperature_values):
                    continue
                self.temperature_index = 0
            finally:
                actual_temperature: Quantity = self.triton.query_temperature(6)
                if not (
                    (1.0 - self.temperature_tolerance) * self.temperature
                    < actual_temperature.to_value(K)
                    < (1.0 + self.temperature_tolerance) * self.temperature
                ):
                    self.temperature_just_set = True

            break

        return False

    def _make_step(self) -> bool:
        self.power_index += 1
        return self._next_indices()

    def on_timeout(self) -> None:
        self._read_state_queue()

        prob: float = np.inf
        err: float
        while not self.results_queue.empty():
            prob, err = self.results_queue.get(block=True)
            self._add_plot_point(self.power_dbm, prob, err)
        self.last_prob = prob

        self._watch_temperature()

        self.good_to_measure.buf[0] &= not self.button_pause.isChecked()
        self.good_to_measure.buf[127] = self.button_drop_measurement.isPushed()

        if not self.measurement.is_alive():
            self.button_drop_measurement.reset()
            self.timer.stop()
            if not self._make_step():
                self.on_button_stop_clicked()
                return

            try:
                self.start_measurement()
            except ConnectionError:
                import traceback

                traceback.print_exc()

                self.on_button_stop_clicked()
                return

        else:
            self.timer.setInterval(50)


if __name__ == "__main__":
    app: QApplication = QApplication(sys.argv)
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    window: App = App()
    window.show()
    app.exec()
    zero_sources()
