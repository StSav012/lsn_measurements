# -*- coding: utf-8 -*-
from __future__ import annotations

import abc
from datetime import date, datetime, timedelta
from multiprocessing import Event, Queue, Value
from pathlib import Path
from typing import Final, Literal, Optional, TextIO

import numpy as np
import pyqtgraph as pg
from astropy.units import K, Quantity
from numpy.typing import NDArray
from qtpy.QtCore import QTimer
from qtpy.QtGui import QCloseEvent, QColor
from qtpy.QtWidgets import QMessageBox

from hardware.anapico import APUASYN20
from hardware.triton import Triton
from measurement.detect import DetectMeasurement
from measurement.lifetime import LifetimeMeasurement
from ui.detect_lifetime_gui import DetectLifetimeGUI
from utils import error, warning
from utils.config import Config
from utils.slice_sequence import SliceSequence
from utils.string_utils import format_float

__all__ = ["DetectLifetimeBase"]


class DetectLifetimeBase(DetectLifetimeGUI):
    def __init__(self) -> None:
        super(DetectLifetimeBase, self).__init__()

        self.timer: QTimer = QTimer(self)
        self.timer.timeout.connect(self.on_timeout)

        self.results_queue_detect: Queue[tuple[float, float]] = Queue()
        self.state_queue_detect: Queue[tuple[int, int, int]] = Queue()
        self.results_queue_lifetime: Queue[tuple[float, float, float]] = Queue()
        self.state_queue_lifetime: Queue[tuple[int, timedelta]] = Queue()
        self.good_to_go: Event = Event()
        self.good_to_go.clear()
        self.user_aborted: Event = Event()
        self.user_aborted.clear()
        self.actual_temperature: Value = Value("d")
        self.measurement: Optional[DetectMeasurement | LifetimeMeasurement] = None

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
        self.r: Final[float] = self.config.get_float("circuitry", "ballast resistance [Ohm]") + self.config.get_float(
            "circuitry",
            "additional ballast resistance [Ohm]",
            fallback=0.0,
        )
        self.r_series: Final[float] = self.config.get_float(
            "circuitry",
            "resistance in series [Ohm]",
            fallback=0.0,
        )

        self.reset_function: Final[str] = self.config.get_str("current", "function", fallback="sine")
        if self.reset_function.casefold() not in ("linear", "half sine", "quarter sine"):
            raise ValueError("Unsupported current reset function:", self.reset_function)
        self.bias_current_values: Final[SliceSequence] = self.config.get_slice_sequence("current", "bias current [nA]")
        self.stop_key_bias.setDisabled(len(self.bias_current_values) <= 1)
        self.initial_biases: Final[list[float]] = self.config.get_float_list(
            "current", "initial current [nA]", fallback=[0.0]
        )
        self.setting_time_values: Final[SliceSequence] = self.config.get_slice_sequence("current", "setting time [sec]")
        self.stop_key_setting_time.setDisabled(len(self.setting_time_values) <= 1)

        self.check_exists: Final[bool] = self.config.get_bool("measurement", "check whether file exists")
        self.trigger_voltage: Final[float] = self.config.get_float("measurement", "voltage trigger [V]") * self.gain
        self.max_reasonable_bias_error: Final[float] = (
            abs(self.config.getfloat("lifetime", "maximal reasonable bias error [%]", fallback=np.inf)) * 0.01
        )
        self.cycles_count_lifetime: Final[int] = self.config.getint("lifetime", "number of cycles")
        self.cycles_count_detect: Final[int] = self.config.getint("detect", "number of cycles")
        self.max_switching_events_count: Final[int] = self.config.getint("detect", "number of switches")
        self.minimal_probability_to_measure: Final[float] = self.config.getfloat(
            "detect", "minimal probability to measure [%]", fallback=0.0
        )
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

        self.frequency_values: Final[SliceSequence] = self.config.get_slice_sequence("GHz signal", "frequency [GHz]")
        self.stop_key_frequency.setDisabled(len(self.frequency_values) <= 1)
        self.power_dbm_values: Final[SliceSequence] = self.config.get_slice_sequence("GHz signal", "power [dBm]")
        self.stop_key_power.setDisabled(len(self.power_dbm_values) <= 1)
        self.pulse_duration: Final[float] = self.config.getfloat("detect", "GHz pulse duration [sec]")
        self.waiting_after_pulse: Final[float] = self.config.getfloat("detect", "waiting after GHz pulse [sec]")

        self.temperature_values: Final[SliceSequence] = self.config.get_slice_sequence("measurement", "temperature")
        self.temperature_delay: Final[timedelta] = timedelta(
            seconds=self.config.getfloat("measurement", "time to wait for temperature [minutes]", fallback=0.0) * 60.0
        )
        self.change_filtered_readings: Final[bool] = self.config.getboolean(
            "measurement", "change filtered readings in Triton", fallback=True
        )
        self.stop_key_temperature.setDisabled(len(self.temperature_values) <= 1)
        self.temperature_tolerance: Final[float] = (
            abs(self.config.getfloat("measurement", "temperature tolerance [%]", fallback=1.0)) * 0.01
        )

        self.saving_location: Path = Path(self.config.get("output", "location", fallback=r"D:\ttt\detect+lifetime"))
        self.saving_location /= self.sample_name
        self.saving_location /= date.today().isoformat()
        self.saving_location.mkdir(parents=True, exist_ok=True)

        self.temperature_index: int = 0
        self.frequency_index: int = 0
        self.setting_time_index: int = 0
        self.delay_between_cycles_index: int = 0
        self.bias_current_index: int = 0
        self.power_index: int = 0

        self.last_lifetime_0: float = np.nan
        self.bad_temperature_time: datetime = datetime.now() - self.temperature_delay
        self.temperature_just_set: bool = False

        self.mode: Literal["detect", "lifetime"] = "detect"

    def closeEvent(self, event: QCloseEvent) -> None:
        self.synthesizer.reset()
        super().closeEvent(event)

    @property
    def temperature(self) -> float:
        return self.temperature_values[self.temperature_index]

    @property
    def bias_current(self) -> float:
        return float(self.bias_current_values[self.bias_current_index])

    @property
    def power_dbm(self) -> float:
        return float(self.power_dbm_values[self.power_index])

    @property
    def frequency(self) -> float:
        return float(self.frequency_values[self.frequency_index])

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
        return {"detect": self.data_file_detect, "lifetime": self.data_file_lifetime}[self.mode]

    @property
    def data_file_detect(self) -> Path:
        return self.saving_location / (
            " ".join(
                filter(
                    None,
                    (
                        "detect-data",
                        self.config.get("output", "prefix", fallback=""),
                        format_float(self.temperature * 1e3, suffix="mK"),
                        format_float(self.bias_current, suffix="nA"),
                        format_float(self.delay_between_cycles, prefix="d", suffix="s"),
                        f"CC{self.cycles_count_detect}",
                        format_float(self.frequency, suffix="GHz") if not np.isnan(self.frequency) else "",
                        format_float(self.power_dbm, suffix="dBm") if not np.isnan(self.power_dbm) else "",
                        format_float(self.pulse_duration, prefix="P", suffix="s"),
                        format_float(self.waiting_after_pulse, prefix="WaP", suffix="s"),
                        format_float(self.setting_time, prefix="ST", suffix="s"),
                        self.config.get("output", "suffix", fallback=""),
                    ),
                )
            )
            + ".txt"
        )

    @property
    def data_file_lifetime(self) -> Path:
        return self.saving_location / (
            " ".join(
                filter(
                    None,
                    (
                        "lifetimes",
                        self.config.get("output", "prefix", fallback=""),
                        format_float(self.temperature * 1e3, suffix="mK"),
                        format_float(self.bias_current, suffix="nA"),
                        format_float(self.delay_between_cycles, prefix="d", suffix="s"),
                        f"CC{self.cycles_count_lifetime}",
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
    def _line_index_detect(self) -> int: ...

    @property
    @abc.abstractmethod
    def _line_name_detect(self) -> str: ...

    @abc.abstractmethod
    def _line_color_detect(self, index: int) -> QColor: ...

    @property
    @abc.abstractmethod
    def _line_index_lifetime(self) -> int: ...

    @property
    @abc.abstractmethod
    def _line_name_lifetime(self) -> str: ...

    @abc.abstractmethod
    def _line_color_lifetime(self, index: int) -> QColor: ...

    @property
    def plot_line_detect(self) -> pg.PlotDataItem:
        i: int = self._line_index_detect
        if i not in self.plot_lines_detect:
            color: QColor = self._line_color_detect(i)
            self.plot_lines_detect[i] = self.canvas_detect.plot(
                np.empty(0, dtype=np.float64),
                symbol="o",
                name=self._line_name_detect or None,
                pen=color,
                symbolPen=color,
                symbolBrush=color,
            )
        return self.plot_lines_detect[i]

    @property
    def plot_line_lifetime(self) -> pg.PlotDataItem:
        i: int = self._line_index_lifetime
        if i not in self.plot_lines_lifetime:
            color: QColor = self._line_color_lifetime(i)
            self.plot_lines_lifetime[i] = self.canvas_lifetime.plot(
                np.empty(0, dtype=np.float64),
                symbol="o",
                name=self._line_name_lifetime or None,
                pen=color,
                symbolPen=color,
                symbolBrush=color,
            )
        return self.plot_lines_lifetime[i]

    def start_measurement_detect(self) -> None:
        if self.measurement is not None and self.measurement.is_alive():
            self.measurement.terminate()
            self.measurement.join()

        self.synthesizer.pulse_modulation.source = "ext"
        self.synthesizer.pulse_modulation.state = True
        self.synthesizer.output = True

        self.measurement = DetectMeasurement(
            results_queue=self.results_queue_detect,
            state_queue=self.state_queue_detect,
            good_to_go=self.good_to_go,
            user_aborted=self.user_aborted,
            actual_temperature=self.actual_temperature,
            resistance=self.r,
            resistance_in_series=self.r_series,
            current_divider=self.divider,
            current_setting_function=self.reset_function,
            initial_biases=self.initial_biases,
            cycles_count=self.cycles_count_detect,
            bias_current=self.bias_current,
            frequency=self.frequency,
            power_dbm=self.power_dbm,
            max_switching_events_count=self.max_switching_events_count,
            pulse_duration=self.pulse_duration,
            setting_time=self.setting_time,
            trigger_voltage=self.trigger_voltage,
            voltage_gain=self.gain,
            temperature=self.temperature,
            stat_file=self.stat_file,
            data_file=self.data_file,
            waiting_after_pulse=self.waiting_after_pulse,
            delay_between_cycles=self.delay_between_cycles,
            adc_rate=self.adc_rate,
        )
        self.measurement.start()

        self.triton.ensure_temperature(6, self.temperature)
        self.label_temperature.setValue(self.temperature * 1000)
        self.label_setting_time.setValue(self.setting_time * 1000)
        self.label_delay_between_cycles.setValue(self.delay_between_cycles * 1000)
        self.synthesizer.frequency = self.frequency * 1e9
        self.label_frequency.setValue(self.frequency)
        self.label_bias.setValue(self.bias_current)
        self.synthesizer.power.level = self.power_dbm
        self.label_power.setValue(self.power_dbm)
        self.label_pulse_duration.setValue(self.pulse_duration * 1000)
        self.label_spent_time.clear()

        self.temperature_just_set = not (
            (1.0 - self.temperature_tolerance) * self.temperature
            < self.triton.query_temperature(6).to_value(K)
            < (1.0 + self.temperature_tolerance) * self.temperature
        )

        print(f"saving to {self.stat_file}")
        self.setWindowTitle(f"Detect+Lifetime — {self.stat_file}")
        self.timer.start(50)

    def start_measurement_lifetime(self) -> None:
        if self.measurement is not None and self.measurement.is_alive():
            self.measurement.terminate()
            self.measurement.join()

        self.synthesizer.power.alc.low_noise = True
        self.synthesizer.output = False

        self.measurement = LifetimeMeasurement(
            results_queue=self.results_queue_lifetime,
            state_queue=self.state_queue_lifetime,
            good_to_go=self.good_to_go,
            user_aborted=self.user_aborted,
            actual_temperature=self.actual_temperature,
            resistance=self.r,
            resistance_in_series=self.r_series,
            current_divider=self.divider,
            current_setting_function=self.reset_function,
            initial_biases=self.initial_biases,
            cycles_count=self.cycles_count_lifetime,
            bias_current=self.bias_current,
            frequency=self.frequency,
            power_dbm=self.power_dbm,
            setting_time=self.setting_time,
            trigger_voltage=self.trigger_voltage,
            voltage_gain=self.gain,
            temperature=self.temperature,
            stat_file=self.stat_file,
            data_file=self.data_file_lifetime,
            ignore_never_switched=self.ignore_never_switched,
            max_waiting_time=self.max_waiting_time,
            max_reasonable_bias_error=self.max_reasonable_bias_error,
            delay_between_cycles=self.delay_between_cycles,
        )
        self.measurement.start()

        self.triton.ensure_temperature(6, self.temperature)
        self.label_temperature.setValue(self.temperature * 1000)
        self.label_setting_time.setValue(self.setting_time * 1000)
        self.label_delay_between_cycles.setValue(self.delay_between_cycles * 1000)
        self.synthesizer.frequency = self.frequency * 1e9
        self.label_frequency.clear()
        self.label_bias.setValue(self.bias_current)
        self.synthesizer.power.level = self.power_dbm
        self.label_power.clear()
        self.label_pulse_duration.clear()
        self.label_loop_count.setValue(self.cycles_count_lifetime)

        self.temperature_just_set = not (
            (1.0 - self.temperature_tolerance) * self.temperature
            < self.triton.query_temperature(6).to_value(K)
            < (1.0 + self.temperature_tolerance) * self.temperature
        )

        print(f"saving to {self.stat_file}")
        self.setWindowTitle(f"Detect+Lifetime — {self.stat_file}")
        self.timer.start(50)

    def start_measurement(self) -> None:
        {
            "detect": self.start_measurement_detect,
            "lifetime": self.start_measurement_lifetime,
        }[self.mode]()

    @abc.abstractmethod
    def _next_indices(self) -> bool: ...

    @abc.abstractmethod
    def _make_step(self) -> bool: ...

    def on_button_start_clicked(self) -> None:
        super(DetectLifetimeBase, self).on_button_start_clicked()

        if self.mode == "detect":
            while self.check_exists and self.stat_file.exists():
                warning(f"{self.stat_file} already exists")
                if not self._next_indices():
                    error("nothing left to measure")
                    self.on_button_stop_clicked()
                    return
        elif self.mode == "lifetime":
            while self.check_exists and self.data_file_lifetime.exists():
                warning(f"{self.data_file_lifetime} already exists")
                if not self._next_indices():
                    error("nothing left to measure")
                    self.on_button_stop_clicked()
                    return

        if self.stat_file.exists():
            f_out: TextIO
            with self.stat_file.open("at", encoding="utf-8") as f_out:
                f_out.write("\n")
        self.start_measurement()

    def on_button_stop_clicked(self) -> None:
        self.user_aborted.set()  # tell the process to finish gracefully
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
        super(DetectLifetimeBase, self).on_button_stop_clicked()

    def _read_state_queue_detect(self) -> None:
        cycle_index: int
        estimated_cycles_count: int
        switches_count: int
        while not self.state_queue_detect.empty():
            (
                cycle_index,
                estimated_cycles_count,
                switches_count,
            ) = self.state_queue_detect.get(block=True)
            self.label_loop_number.setValue(cycle_index + 1)
            self.label_loop_count.setValue(estimated_cycles_count)
            self.label_probability.setValue(switches_count / (cycle_index + 1) * 100)

    def _add_plot_point_detect(self, x: float, prob: float, err: float) -> None:
        old_x_data: NDArray[np.float64] = (
            np.empty(0, dtype=np.float64) if self.plot_line_detect.xData is None else self.plot_line_detect.xData
        )
        old_y_data: NDArray[np.float64] = (
            np.empty(0, dtype=np.float64) if self.plot_line_detect.yData is None else self.plot_line_detect.yData
        )
        x_data: NDArray[np.float64] = np.append(old_x_data, x)
        y_data: NDArray[np.float64] = np.append(old_y_data, prob)
        self.plot_line_detect.setData(x_data, y_data)

    def _read_state_queue_lifetime(self) -> None:
        cycle_index: int
        spent_time: timedelta
        while not self.state_queue_lifetime.empty():
            cycle_index, spent_time = self.state_queue_lifetime.get(block=True)
            self.label_loop_number.setValue(cycle_index + 1)
            self.label_spent_time.setValue(spent_time.total_seconds())

    def _add_plot_point_lifetime(self, x: float, lifetime: float) -> None:
        old_x_data: NDArray[np.float64] = (
            np.empty(0, dtype=np.float64) if self.plot_line_lifetime.xData is None else self.plot_line_lifetime.xData
        )
        old_y_data: NDArray[np.float64] = (
            np.empty(0, dtype=np.float64) if self.plot_line_lifetime.yData is None else self.plot_line_lifetime.yData
        )
        x_data: NDArray[np.float64] = np.append(old_x_data, x)
        y_data: NDArray[np.float64] = np.append(old_y_data, lifetime)
        self.plot_line_lifetime.setData(x_data, y_data)

    def _is_temperature_good(self) -> bool:
        td: timedelta
        actual_temperature: Quantity = self.triton.query_temperature(6)
        self.actual_temperature.value = actual_temperature.to_value("mK")
        good_to_go: bool
        if not (
            (1.0 - self.temperature_tolerance) * self.temperature
            < actual_temperature.to_value(K)
            < (1.0 + self.temperature_tolerance) * self.temperature
        ):
            good_to_go = False
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
                good_to_go = True
                self.temperature_just_set = False
            else:
                good_to_go = False
                print(
                    f"temperature {actual_temperature} "
                    f"is close enough to {self.temperature:.3f} K, but not for long enough yet"
                    f": {self.temperature_delay - td} left"
                )
                self.timer.setInterval(1000)
        else:
            good_to_go = True

        return good_to_go

    def _stat_file_exists(self, verbose: bool = True) -> bool:
        exists: bool = (
            self.bias_current_index < len(self.bias_current_values)
            and self.power_index < len(self.power_dbm_values)
            and self.frequency_index < len(self.frequency_values)
            and self.setting_time_index < len(self.setting_time_values)
            and self.delay_between_cycles_index < len(self.delay_between_cycles_values)
            and self.temperature_index < len(self.temperature_values)
            and self.stat_file.exists()
        )
        if exists and verbose:
            warning(f"{self.stat_file} already exists")
        return exists

    @abc.abstractmethod
    def on_timeout(self) -> None: ...
