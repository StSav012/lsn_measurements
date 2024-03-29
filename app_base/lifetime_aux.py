# -*- coding: utf-8 -*-
from __future__ import annotations

import abc
from datetime import date, datetime, timedelta
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Final, TextIO

import numpy as np
import pyqtgraph as pg
from astropy.units import K, Quantity
from numpy.typing import NDArray
from qtpy.QtCore import QTimer
from qtpy.QtGui import QCloseEvent, QColor
from qtpy.QtWidgets import QMessageBox

from hardware.anapico import APUASYN20
from hardware.triton import Triton
from measurement.lifetime_aux import LifetimeMeasurement
from ui.lifetime_aux_gui import LifetimeGUI
from utils import error, warning
from utils.config import Config
from utils.slice_sequence import SliceSequence
from utils.string_utils import format_float

__all__ = ["LifetimeBase"]


class LifetimeBase(LifetimeGUI):
    def __init__(self) -> None:
        super(LifetimeBase, self).__init__()

        self.timer: QTimer = QTimer(self)
        self.timer.timeout.connect(self.on_timeout)

        self.results_queue: Queue[tuple[float, float, float]] = Queue()
        self.state_queue: Queue[tuple[int, timedelta]] = Queue()
        self.good_to_measure: SharedMemory = SharedMemory(create=True, size=128)
        self.good_to_measure.buf[0] = False
        self.measurement: LifetimeMeasurement | None = None

        self.config: Config = Config()

        try:
            self.triton: Triton = Triton()
        except Exception as ex:
            QMessageBox.critical(
                self,
                ex.__class__.__name__,
                "\n".join((self.tr("Failed to initialize Triton"), str(ex))),
            )
            raise

        self.triton.query_temperature(6, blocking=True)

        self.synthesizer: APUASYN20 = APUASYN20(expected=self.config.getboolean("GHz signal", "connect", fallback=True))

        self.sample_name: Final[str] = self.config.sample_name
        self.parameters_box.setTitle(self.sample_name)
        self.gain: Final[float] = self.config.get_float("circuitry", "voltage gain")
        self.divider: Final[float] = self.config.get_float("circuitry", "current divider")
        self.r: Final[float] = self.config.get_float(
            "circuitry",
            "ballast resistance [Ohm]",
        ) + self.config.get_float(
            "circuitry",
            "additional ballast resistance [Ohm]",
            fallback=0.0,
        )
        self.r_series: Final[float] = self.config.get_float("circuitry", "resistance in series [Ohm]", fallback=0.0)

        self.reset_function: Final[str] = self.config.get_str("current", "function")
        if self.reset_function.casefold() not in ("linear", "half sine", "quarter sine"):
            raise ValueError("Unsupported current reset function:", self.reset_function)
        self.bias_current_values: SliceSequence = self.config.get_slice_sequence("current", "bias current [nA]")
        self.stop_key_bias.setDisabled(len(self.bias_current_values) <= 1)
        self.initial_biases: Final[list[float]] = self.config.get_float_list(
            "current", "initial current [nA]", fallback=[0.0]
        )

        self.setting_time_values: Final[SliceSequence] = self.config.get_slice_sequence("current", "setting time [sec]")
        self.stop_key_setting_time.setDisabled(len(self.setting_time_values) <= 1)

        self.check_exists: Final[bool] = self.config.getboolean("measurement", "check whether file exists")
        self.trigger_voltage: Final[float] = self.config.get_float("measurement", "voltage trigger [V]") * self.gain
        self.max_reasonable_bias_error: Final[float] = (
            abs(self.config.getfloat("lifetime", "maximal reasonable bias error [%]", fallback=np.inf)) * 0.01
        )
        self.cycles_count: Final[int] = self.config.getint("lifetime", "number of cycles")
        self.max_waiting_time: Final[timedelta] = timedelta(
            seconds=self.config.getfloat("lifetime", "max time of waiting for switching [sec]")
        )
        self.max_mean: Final[float] = self.config.getfloat(
            "lifetime", "max mean time to measure [sec]", fallback=np.inf
        )
        self.ignore_never_switched: Final[bool] = self.config.getboolean("lifetime", "ignore never switched")
        self.delay_between_cycles_values: Final[SliceSequence] = self.config.get_slice_sequence(
            "measurement", "delay between cycles [sec]"
        )
        self.stop_key_delay_between_cycles.setDisabled(len(self.delay_between_cycles_values) <= 1)
        self.adc_rate: Final[float] = self.config.get_float("measurement", "adc rate [S/sec]", fallback=np.nan)

        self.synthesizer_output: Final[bool] = self.config.getboolean("GHz signal", "on", fallback=False)
        self.frequency_values: Final[SliceSequence] = (
            self.config.get_slice_sequence("GHz signal", "frequency [GHz]", fallback=SliceSequence())
            if self.synthesizer_output
            else SliceSequence()
        )
        self.stop_key_frequency.setDisabled(len(self.frequency_values) <= 1)
        self.power_dbm_values: Final[SliceSequence] = (
            self.config.get_slice_sequence("GHz signal", "power [dBm]", fallback=SliceSequence())
            if self.synthesizer_output
            else SliceSequence()
        )
        self.stop_key_power.setDisabled(len(self.power_dbm_values) <= 1)

        self.aux_voltage_values: Final[SliceSequence] = self.config.get_slice_sequence("measurement", "aux voltage [V]")
        self.aux_voltage_delay: Final[timedelta] = timedelta(
            seconds=self.config.getfloat(
                "measurement", "time to wait after aux voltage changed [minutes]", fallback=0.0
            )
            * 60.0
        )
        self.stop_key_aux_voltage.setDisabled(len(self.aux_voltage_values) <= 1)

        self.temperature_values: Final[SliceSequence] = self.config.get_slice_sequence("measurement", "temperature")
        self.temperature_delay: Final[timedelta] = timedelta(
            seconds=self.config.get_float("measurement", "time to wait for temperature [minutes]", fallback=0.0) * 60.0
        )
        self.stop_key_temperature.setDisabled(len(self.temperature_values) <= 1)
        self.temperature_tolerance: Final[float] = (
            abs(self.config.get_float("measurement", "temperature tolerance [%]", fallback=0.5)) * 0.01
        )
        self.change_filtered_readings: Final[bool] = self.config.getboolean(
            "measurement", "change filtered readings in Triton", fallback=True
        )

        self.saving_location: Path = Path(self.config.get("output", "location", fallback=r"D:\ttt\lifetime"))
        self.saving_location /= self.sample_name
        self.saving_location /= date.today().isoformat()
        self.saving_location.mkdir(parents=True, exist_ok=True)

        self.temperature_index: int = 0
        self.aux_voltage_index: int = 0
        self.frequency_index: int = 0
        self.setting_time_index: int = 0
        self.delay_between_cycles_index: int = 0
        self.bias_current_index: int = 0
        self.power_index: int = 0

        self.saved_files: set[Path] = set()

        self.loop_data: dict[int, timedelta] = dict()
        self.last_lifetime_0: float = np.nan
        self.bad_temperature_time: datetime = datetime.now() - self.temperature_delay
        self.bad_aux_voltage_time: datetime = datetime.now()
        self.temperature_just_set: bool = False

    def closeEvent(self, event: QCloseEvent) -> None:
        self.synthesizer.reset()
        super().closeEvent(event)

    @property
    def temperature(self) -> float:
        return self.temperature_values[self.temperature_index]

    @property
    def aux_voltage(self) -> float:
        return self.aux_voltage_values[self.aux_voltage_index]

    @property
    def bias_current(self) -> float:
        return float(self.bias_current_values[self.bias_current_index])

    @property
    def power_dbm(self) -> float:
        return (
            float(self.power_dbm_values[self.power_index]) if self.power_index < len(self.power_dbm_values) else np.nan
        )

    @property
    def frequency(self) -> float:
        return (
            float(self.frequency_values[self.frequency_index])
            if self.frequency_index < len(self.frequency_values)
            else np.nan
        )

    @property
    def setting_time(self) -> float:
        return self.setting_time_values[self.setting_time_index]

    @property
    def delay_between_cycles(self) -> float:
        return self.delay_between_cycles_values[self.delay_between_cycles_index]

    @property
    @abc.abstractmethod
    def stat_file(self) -> Path: ...

    @property
    def data_file(self) -> Path:
        return self.saving_location / (
            " ".join(
                filter(
                    None,
                    (
                        "lifetimes",
                        self.config.get("output", "prefix", fallback=""),
                        format_float(self.temperature * 1e3, suffix="mK"),
                        format_float(self.aux_voltage * 1e3, prefix="aux", suffix="mV"),
                        format_float(self.bias_current, suffix="nA"),
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

    @property
    def hist_file(self) -> Path:
        return self.saving_location / (
            " ".join(
                filter(
                    None,
                    (
                        "lifetimes-hist",
                        self.config.get("output", "prefix", fallback=""),
                        format_float(self.temperature * 1e3, suffix="mK"),
                        format_float(self.aux_voltage * 1e3, prefix="aux", suffix="mV"),
                        format_float(self.bias_current, suffix="nA"),
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

    @property
    @abc.abstractmethod
    def _line_index(self) -> int: ...

    @property
    @abc.abstractmethod
    def _line_name(self) -> str: ...

    @abc.abstractmethod
    def _line_color(self, index: int) -> QColor: ...

    @property
    def plot_line(self) -> pg.PlotDataItem:
        i: int = self._line_index
        if i not in self.plot_lines:
            color: QColor = self._line_color(i)
            self.plot_lines[i] = self.figure.plot(
                np.empty(0, dtype=np.float64),
                symbol="o",
                name=self._line_name or None,
                pen=color,
                symbolPen=color,
                symbolBrush=color,
            )
        return self.plot_lines[i]

    def start_measurement(self) -> None:
        if self.measurement is not None and self.measurement.is_alive():
            self.measurement.terminate()
            self.measurement.join()

        self.loop_data.clear()

        self.synthesizer.output = self.synthesizer_output
        self.synthesizer.power.alc.low_noise = True

        self.measurement = LifetimeMeasurement(
            results_queue=self.results_queue,
            state_queue=self.state_queue,
            good_to_go=self.good_to_measure,
            resistance=self.r,
            resistance_in_series=self.r_series,
            current_divider=self.divider,
            current_setting_function=self.reset_function,
            initial_biases=self.initial_biases,
            cycles_count=self.cycles_count,
            bias_current=self.bias_current,
            aux_voltage=self.aux_voltage,
            frequency=self.frequency,
            power_dbm=self.power_dbm,
            setting_time=self.setting_time,
            trigger_voltage=self.trigger_voltage,
            voltage_gain=self.gain,
            temperature=self.temperature,
            stat_file=self.stat_file,
            data_file=self.data_file,
            ignore_never_switched=self.ignore_never_switched,
            max_waiting_time=self.max_waiting_time,
            max_reasonable_bias_error=self.max_reasonable_bias_error,
            delay_between_cycles=self.delay_between_cycles,
            adc_rate=self.adc_rate,
        )
        self.measurement.start()

        self.triton.ensure_temperature(6, self.temperature)
        self.label_temperature.setValue(self.temperature * 1000)
        self.label_aux_voltage.setValue(self.aux_voltage * 1000)
        self.label_setting_time.setValue(self.setting_time * 1000)
        self.label_delay_between_cycles.setValue(self.delay_between_cycles * 1000)
        self.synthesizer.frequency = self.frequency * 1e9
        self.label_frequency.setValue(self.frequency)
        self.label_bias.setValue(self.bias_current)
        self.synthesizer.power.level = self.power_dbm
        self.label_power.setValue(self.power_dbm)

        self.temperature_just_set = not (
            (1.0 - self.temperature_tolerance) * self.temperature
            < self.triton.query_temperature(6).to_value(K)
            < (1.0 + self.temperature_tolerance) * self.temperature
        )

        self.button_drop_measurement.reset()

        print(f"saving to {self.stat_file}")
        self.setWindowTitle(f"Lifetime — {self.stat_file}")
        self.timer.start(50)

    @abc.abstractmethod
    def _next_indices(self, make_step: bool = True) -> bool: ...

    def on_button_start_clicked(self) -> None:
        super(LifetimeBase, self).on_button_start_clicked()
        if self.check_exists and not self._next_indices(make_step=False):
            error("nothing left to measure")
            self.on_button_stop_clicked()
            return

        if self.stat_file.exists():
            f_out: TextIO
            with self.stat_file.open("at", encoding="utf-8") as f_out:
                f_out.write("\n")
        self.start_measurement()

    def on_button_stop_clicked(self) -> None:
        self.good_to_measure.buf[127] = True  # tell the process to finish gracefully
        if self.measurement is not None:
            if self.measurement.is_alive():
                try:
                    self.measurement.join(1)
                except TimeoutError:  # still alive
                    self.measurement.terminate()
                    self.measurement.join()
            else:
                self.measurement.join()
        self.timer.stop()
        self.synthesizer.output = False
        self.saved_files.add(self.data_file)
        self.histogram.save(self.hist_file)
        super(LifetimeBase, self).on_button_stop_clicked()

    def _read_state_queue(self) -> None:
        cycle_index: int
        spent_time: timedelta
        while not self.state_queue.empty():
            cycle_index, spent_time = self.state_queue.get(block=True)
            self.label_loop_number.setValue(cycle_index + 1)
            self.label_spent_time.setValue(spent_time.total_seconds())
            self.loop_data[cycle_index] = spent_time
        finished_data: NDArray[np.float64] = np.asarray([v.total_seconds() for v in self.loop_data.values()][:-1])
        self.histogram.hist(finished_data, pen="white", symbolBrush="white", symbolPen="white")
        if finished_data.size:
            self.label_mean_lifetime.setValue(np.mean(finished_data))
        else:
            self.label_mean_lifetime.clear()
        if finished_data.size > 2:
            self.label_lifetime_std.setValue(np.std(finished_data))
            self.label_lifetime_mean_std_ratio.setValue(np.mean(finished_data) / np.std(finished_data))
        else:
            self.label_lifetime_std.clear()
            self.label_lifetime_mean_std_ratio.clear()

    def _add_plot_point(self, x: float, lifetime: float) -> None:
        old_x_data: NDArray[np.float64] = (
            np.empty(0, dtype=np.float64) if self.plot_line.xData is None else self.plot_line.xData
        )
        old_y_data: NDArray[np.float64] = (
            np.empty(0, dtype=np.float64) if self.plot_line.yData is None else self.plot_line.yData
        )
        x_data: NDArray[np.float64] = np.append(old_x_data, x)
        y_data: NDArray[np.float64] = np.append(old_y_data, lifetime)
        self.plot_line.setData(x_data, y_data)

    def _watch_temperature(self) -> None:
        td: timedelta
        actual_temperature: Quantity = self.triton.query_temperature(6)
        ats: bytes = str(actual_temperature.to_value("mK")).encode()
        self.good_to_measure.buf[1 : 1 + len(ats)] = ats
        if not (
            (1.0 - self.temperature_tolerance) * self.temperature
            < actual_temperature.to_value(K)
            < (1.0 + self.temperature_tolerance) * self.temperature
        ):
            self.good_to_measure.buf[0] = False
            self.bad_temperature_time = datetime.now()
            self.timer.setInterval(1000)
            print(f"temperature {actual_temperature} is too far from {self.temperature:.3f} K")
            if not self.triton.ensure_temperature(6, self.temperature):
                error(f"failed to set temperature to {self.temperature} K")
                self.timer.stop()
                self.measurement.terminate()
            if self.change_filtered_readings:
                if not self.triton.ensure_filter_readings(6, self.triton.filter_readings(self.temperature)):
                    error("failed to change the state of filtered readings")
                    self.timer.stop()
                    self.measurement.terminate()
            if not self.triton.ensure_heater_range(6, self.triton.heater_range(self.temperature)):
                error("failed to change the heater range")
                self.timer.stop()
                self.measurement.terminate()
        elif self.temperature_just_set:
            td = datetime.now() - self.bad_temperature_time
            if td > self.temperature_delay:
                self.timer.setInterval(50)
                self.good_to_measure.buf[0] = True
                self.temperature_just_set = False
            else:
                self.good_to_measure.buf[0] = False
                print(
                    f"temperature {actual_temperature} "
                    f"is close enough to {self.temperature:.3f} K, but not for long enough yet"
                    f": {self.temperature_delay - td} left"
                )
                self.timer.setInterval(1000)
        else:
            self.good_to_measure.buf[0] = True

        if self.good_to_measure.buf[0]:
            td = datetime.now() - self.bad_aux_voltage_time
            if td <= self.aux_voltage_delay:
                self.good_to_measure.buf[0] = False
                print(f"wait for {self.aux_voltage_delay - td} after the aux voltage change")

    def _data_file_exists(self, verbose: bool = True) -> bool:
        exists: bool = (
            self.bias_current_index < len(self.bias_current_values)
            and self.power_index < len(self.power_dbm_values)
            and self.frequency_index < len(self.frequency_values)
            and self.setting_time_index < len(self.setting_time_values)
            and self.delay_between_cycles_index < len(self.delay_between_cycles_values)
            and self.aux_voltage_index < len(self.aux_voltage_values)
            and self.temperature_index < len(self.temperature_values)
            and self.data_file.exists()
            and self._get_data_file_content().size
        )
        if exists and verbose:
            if self.data_file not in self.saved_files:
                warning(f"{self.data_file} already exists")
        return exists

    def _get_data_file_content(self) -> NDArray[float]:
        return np.array(
            [
                [float(cell) for cell in row.split("\t")]
                for row in self.data_file.read_text(encoding="utf-8").splitlines()
                if row and (row.startswith("nan") or not row[0].isalpha())
            ]
        ).T

    @abc.abstractmethod
    def on_timeout(self) -> None: ...
