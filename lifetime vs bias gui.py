# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from typing import cast, final

import numpy as np
from astropy.units import K, Quantity
from numpy.typing import NDArray
from pyqtgraph.functions import intColor
from qtpy import QT5
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QApplication

from app_base.lifetime import LifetimeBase
from hardware import dac_aux
from utils.ni import zero_sources
from utils.string_utils import format_float


@final
class App(LifetimeBase):
    def setup_ui_appearance(self) -> None:
        super(App, self).setup_ui_appearance()

        self.figure.getAxis("bottom").setLabel(text=self.tr("Current"), units=self.tr("nA"))

    @property
    def stat_file(self) -> Path:
        return self.saving_location / (
            " ".join(
                filter(
                    None,
                    (
                        "lifetime",
                        self.config.get("output", "prefix", fallback=""),
                        format_float(self.temperature * 1e3, suffix="mK"),
                        format_float(self.bias_current, suffix="nA") if len(self.bias_current_values) == 1 else "",
                        format_float(self.delay_between_cycles, prefix="d", suffix="s"),
                        f"CC{self.cycles_count}",
                        format_float(self.setting_time, prefix="ST", suffix="s"),
                        format_float(self.frequency, suffix="GHz") if not np.isnan(self.frequency) else "",
                        format_float(self.power_dbm, suffix="dBm") if not np.isnan(self.power_dbm) else "",
                        format_float(self.initial_biases[-1], prefix="from ", suffix="nA"),
                        self.config.get("output", "suffix", fallback=""),
                    ),
                )
            )
            + ".txt"
        )

    # fmt: off
    @property
    def _line_index(self) -> int:
        return (self.power_index
                + (self.frequency_index
                   + (self.setting_time_index
                      + (self.delay_between_cycles_index
                         + (self.temperature_index
                            ) * (len(self.delay_between_cycles_values) or 1)
                         ) * (len(self.setting_time_values) or 1)
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
                    (
                        format_float(self.temperature * 1e3, suffix=self.tr("mK"))
                        if len(self.temperature_values) > 1
                        else ""
                    ),
                    format_float(self.bias_current, suffix=self.tr("nA")) if len(self.bias_current_values) == 1 else "",
                    (
                        format_float(self.setting_time * 1e3, prefix=self.tr("ST "), suffix=self.tr("ms"))
                        if len(self.setting_time_values) > 1
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
            hues *= len(self.frequency_values) or 1
        if hues < 7:
            hues *= len(self.setting_time_values) or 1
        if hues < 7:
            hues *= len(self.delay_between_cycles_values) or 1
        if hues < 7:
            hues *= len(self.temperature_values) or 1
        return intColor(index, hues=hues)

    def _add_plot_point_from_file(self) -> None:
        if self.data_file in self.saved_files:
            return
        self.saved_files.add(self.data_file)
        measured_data: NDArray[float] = self._get_data_file_content()
        if measured_data.shape[0] == 7:
            bias_current: NDArray[float] = measured_data[3]
            lifetime: NDArray[float] = measured_data[2]
            median_bias_current: float = cast(float, np.nanmedian(bias_current))
            min_reasonable_bias_current: float = median_bias_current * (1.0 - self.max_reasonable_bias_error)
            max_reasonable_bias_current: float = median_bias_current * (1.0 + self.max_reasonable_bias_error)
            reasonable: NDArray[np.bool_] = (bias_current >= min_reasonable_bias_current) & (
                bias_current <= max_reasonable_bias_current
            )
            bias_current = bias_current[reasonable]
            lifetime = lifetime[reasonable]
            self._add_plot_point(
                cast(float, np.mean(bias_current)),
                cast(float, np.mean(lifetime[lifetime > 0.0])),
            )
            self.last_lifetime_0 = cast(float, np.mean(lifetime[lifetime > 0.0]))

    def _next_indices(self) -> bool:
        while True:
            if self.stop_key_bias.isChecked():
                return False
            while self.check_exists and self._data_file_exists():
                self._add_plot_point_from_file()
                self.bias_current_index += 1
            if (
                np.isnan(self.last_lifetime_0) or self.last_lifetime_0 <= self.max_mean
            ) and self.bias_current_index < len(self.bias_current_values):
                return True
            self.bias_current_index = 0

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

            if self.stop_key_setting_time.isChecked():
                return False
            self.setting_time_index += 1
            if self.setting_time_index < len(self.setting_time_values):
                continue
            self.setting_time_index = 0

            if self.stop_key_delay_between_cycles.isChecked():
                return False
            self.delay_between_cycles_index += 1
            if self.delay_between_cycles_index < len(self.delay_between_cycles_values):
                continue
            self.delay_between_cycles_index = 0

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
        self.bias_current_index += 1
        return self._next_indices()

    def on_timeout(self) -> None:
        self._read_state_queue()

        set_bias: float
        lifetime_0: float
        lifetime: float
        while not self.results_queue.empty():
            set_bias, lifetime_0, lifetime = self.results_queue.get(block=True)
            self._add_plot_point(set_bias, lifetime)
            self.last_lifetime_0 = lifetime_0

        if self._is_temperature_good() and not self.button_pause.isChecked():
            self.good_to_go.set()
        else:
            self.good_to_go.clear()
        if self.button_drop_measurement.isPushed():
            self.user_aborted.set()
        else:
            self.user_aborted.clear()

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
    if QT5:
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    window: App = App()
    window.show()
    app.exec()
    zero_sources(reset_dac=False, exceptions=(dac_aux,))
