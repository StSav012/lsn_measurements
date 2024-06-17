# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from typing import final

import numpy as np
from astropy.units import K, Quantity
from pyqtgraph.functions import intColor
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QApplication

from app_base.scd import SwitchingCurrentDistributionBase
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
                    format_float(self.frequency, suffix="GHz") if not np.isnan(self.frequency) else "",
                    format_float(self.power_dbm, suffix="dBm") if not np.isnan(self.power_dbm) else "",
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
        return (self.current_speed_index
                + (self.delay_between_cycles_index
                   + (self.power_index
                      + (self.frequency_index
                         ) * (len(self.power_dbm_values) or 1)
                      ) * (len(self.delay_between_cycles_values) or 1)
                   ) * (len(self.current_speed_values) or 1)
                )
    # fmt: on

    @property
    def _line_name(self) -> str:
        return ", ".join(
            filter(
                None,
                (
                    (
                        format_float(self.temperature * 1e3, suffix=self.tr("mK"))
                        if len(self.temperature_values) == 1
                        else ""
                    ),
                    (
                        format_float(self.current_speed, suffix=self.tr("nA/s"))
                        if len(self.current_speed_values) > 1
                        else ""
                    ),
                    (
                        format_float(self.delay_between_cycles * 1e3, prefix=self.tr("d "), suffix=self.tr("ms"))
                        if len(self.delay_between_cycles_values) > 1
                        else ""
                    ),
                    (
                        format_float(self.frequency, suffix=self.tr("GHz"))
                        if len(self.frequency_values) > 1 and not np.isnan(self.frequency)
                        else ""
                    ),
                    (
                        format_float(self.power_dbm, suffix=self.tr("dBm"))
                        if len(self.power_dbm_values) > 1 and not np.isnan(self.power_dbm)
                        else ""
                    ),
                ),
            )
        )

    def _line_color(self, index: int) -> QColor:
        hues: int = len(self.power_dbm_values) or 1
        if hues < 7:
            hues *= len(self.current_speed_values) or 1
        if hues < 7:
            hues *= len(self.delay_between_cycles_values) or 1
        if hues < 7:
            hues *= len(self.frequency_values) or 1
        return intColor(index, hues=hues)

    def _next_indices(self) -> bool:
        while True:
            if self.stop_key_current_speed.isChecked():
                return False
            while self.check_exists and self._data_file_exists():
                self._add_plot_point_from_file(self.temperature)
                self.current_speed_index += 1
            if self.current_speed_index < len(self.current_speed_values):
                return True
            self.current_speed_index = 0

            if self.stop_key_delay_between_cycles.isChecked():
                return False
            self.delay_between_cycles_index += 1
            if self.delay_between_cycles_index < len(self.delay_between_cycles_values):
                continue
            self.delay_between_cycles_index = 0

            if self.stop_key_power.isChecked():
                return False
            self.power_index += 1
            if self.power_index < len(self.power_dbm_values):
                continue
            self.power_index = 0

            if self.stop_key_frequency.isChecked():
                return False
            self.frequency_index += 1
            if self.frequency_index < len(self.frequency_values):
                continue
            self.frequency_index = 0

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
        self.current_speed_index += 1
        return self._next_indices()

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
            if not self._make_step():
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
