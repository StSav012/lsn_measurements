# -*- coding: utf-8 -*-
import sys
from datetime import datetime
from pathlib import Path
from typing import final

import numpy as np
from pyqtgraph.functions import intColor
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QApplication

from app_base.scd_aux import SwitchingCurrentDistributionBase
from hardware import dac_aux
from utils.ni import zero_sources
from utils.string_utils import format_float


@final
class App(SwitchingCurrentDistributionBase):
    def setup_ui_appearance(self) -> None:
        super(App, self).setup_ui_appearance()

        self.canvas_mean.getAxis("bottom").setLabel(text=self.tr("Temperature"), units=self.tr("K"))
        self.canvas_std.getAxis("bottom").setLabel(text=self.tr("Temperature"), units=self.tr("K"))
        self.canvas_mean.getAxis("bottom").enableAutoSIPrefix(True)
        self.canvas_std.getAxis("bottom").enableAutoSIPrefix(True)

    @property
    def stat_file(self) -> Path:
        return self.saving_location / (
            " ".join(
                (
                    "SCD-stat",
                    self.config.get("output", "prefix", fallback=""),
                    format_float(self.temperature * 1e3, suffix="mK") if len(self.temperature_values) == 1 else "",
                    format_float(self.current_speed, prefix="v", suffix="nAps"),
                    format_float(self.delay_between_cycles, prefix="d", suffix="s"),
                    f"CC{self.cycles_count}",
                    format_float(self.frequency, suffix="GHz") if self.synthesizer_output else "",
                    format_float(self.power_dbm, suffix="dBm") if self.synthesizer_output else "",
                    format_float(self.initial_biases[-1], prefix="from ", suffix="nA"),
                    format_float(self.trigger_voltage * 1e3, prefix="threshold", suffix="mV"),
                    self.config.get("output", "suffix", fallback=""),
                )
            )
            .replace("  ", " ")
            .replace("  ", " ")
            .strip(" ")
            + ".txt"
        )

    # fmt: off
    @property
    def _line_index(self) -> int:
        return (self.power_index
                + (self.frequency_index
                   + (self.current_speed_index
                      + (self.delay_between_cycles_index
                         + (self.aux_voltage_index
                            ) * (len(self.delay_between_cycles_values) or 1)
                         ) * (len(self.current_speed_values) or 1)
                      ) * (len(self.frequency_values) or 1)
                   ) * (len(self.power_dbm_values) or 1)
                )
    # fmt: on

    @property
    def _line_name(self) -> str:
        return ", ".join(
            filter(
                None,
                (
                    format_float(self.aux_voltage * 1e3, suffix=self.tr("mV")),
                    format_float(self.current_speed, suffix=self.tr("nA/s")),
                    format_float(
                        self.delay_between_cycles * 1e3,
                        prefix="d ",
                        suffix=self.tr("ms"),
                    ),
                    format_float(self.power_dbm, suffix=self.tr("dBm")) if not np.isnan(self.power_dbm) else "",
                    format_float(self.frequency, suffix=self.tr("GHz")) if not np.isnan(self.frequency) else "",
                ),
            )
        )

    def _line_color(self, index: int) -> QColor:
        hues: int = len(self.power_dbm_values) or 1
        if hues < 7:
            hues *= len(self.frequency_values) or 1
        if hues < 7:
            hues *= len(self.current_speed_values) or 1
        if hues < 7:
            hues *= len(self.delay_between_cycles_values) or 1
        if hues < 7:
            hues *= len(self.aux_voltage_values) or 1
        return intColor(index, hues=hues)

    def _next_indices(self, make_step: bool = True) -> bool:
        if self.stop_key_power.isChecked():
            return False
        if make_step:
            self.power_index += 1
        while self.check_exists and self._data_file_exists():
            self._add_plot_point_from_file(self.aux_voltage)
            self.power_index += 1
        if self.power_index >= len(self.power_dbm_values):
            self.power_index = 0
            if self.stop_key_frequency.isChecked():
                return False
            if make_step:
                self.frequency_index += 1
            while self.check_exists and self._data_file_exists():
                self._add_plot_point_from_file(self.aux_voltage)
                self.frequency_index += 1
            if self.frequency_index >= len(self.frequency_values):
                self.frequency_index = 0
                if self.stop_key_current_speed.isChecked():
                    return False
                if make_step:
                    self.current_speed_index += 1
                while self.check_exists and self._data_file_exists():
                    self._add_plot_point_from_file(self.aux_voltage)
                    self.current_speed_index += 1
                if self.current_speed_index >= len(self.current_speed_values):
                    self.current_speed_index = 0
                    if self.stop_key_delay_between_cycles.isChecked():
                        return False
                    if make_step:
                        self.delay_between_cycles_index += 1
                    while self.check_exists and self._data_file_exists():
                        self._add_plot_point_from_file(self.aux_voltage)
                        self.delay_between_cycles_index += 1
                    if self.delay_between_cycles_index >= len(self.delay_between_cycles_values):
                        self.delay_between_cycles_index = 0
                        if self.stop_key_aux_voltage.isChecked():
                            return False
                        if make_step:
                            self.aux_voltage_index += 1
                            self.bad_aux_voltage_time = datetime.now()
                        while self.check_exists and self._data_file_exists():
                            self._add_plot_point_from_file(self.aux_voltage)
                            self.aux_voltage_index += 1
                            self.bad_aux_voltage_time = datetime.now()
                        if self.aux_voltage_index >= len(self.aux_voltage_values):
                            self.aux_voltage_index = 0
                            self.bad_aux_voltage_time = datetime.now()
                            if self.stop_key_temperature.isChecked():
                                return False
                            if make_step:
                                self.temperature_index += 1
                            while self.check_exists and self._data_file_exists():
                                self._add_plot_point_from_file(self.aux_voltage)
                                self.temperature_index += 1
                            if self.temperature_index >= len(self.temperature_values):
                                self.temperature_index = 0
                                return False
                        actual_temperature: float
                        temperature_unit: str
                        (
                            actual_temperature,
                            temperature_unit,
                        ) = self.triton.query_temperature(6)
                        if not (
                            (1.0 - self.temperature_tolerance) * self.temperature
                            < actual_temperature
                            < (1.0 + self.temperature_tolerance) * self.temperature
                        ):
                            self.temperature_just_set = True
        return True

    def on_timeout(self) -> None:
        self._read_state_queue()
        self._read_switching_data_queue()

        mean: float
        std: float
        while not self.results_queue.empty():
            mean, std = self.results_queue.get(block=True)
            self._add_plot_point(self.temperature, mean, std)

        self._watch_temperature()

        self.good_to_measure.buf[0] &= not self.button_pause.isChecked()
        self.good_to_measure.buf[127] = self.button_drop_measurement.isPushed()

        if not self.measurement.is_alive():
            self.button_drop_measurement.reset()
            self.timer.stop()
            self.histogram.save(self.hist_file)
            if not self._next_indices():
                self.on_button_stop_clicked()
                return

            self.start_measurement()
        else:
            self.timer.setInterval(50)


if __name__ == "__main__":
    app: QApplication = QApplication(sys.argv)
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    window: App = App()
    window.show()
    app.exec()
    zero_sources(reset_dac=False, exceptions=(dac_aux,))
