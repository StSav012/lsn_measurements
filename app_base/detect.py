# -*- coding: utf-8 -*-
from __future__ import annotations

import abc
from configparser import ConfigParser
from datetime import date, datetime, timedelta
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Final

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from qtpy.QtCore import QTimer
from qtpy.QtGui import QCloseEvent, QColor

from backend.communication.anapico_communication import APUASYN20
from backend.communication.triton_communication import Triton
from backend.measurement.detect import DetectMeasurement
from backend.utils import SliceSequence, error, warning
from backend.utils.config import get_float, get_float_list, get_str
from backend.utils.string_utils import format_float
from ui.detect_gui import DetectGUI

__all__ = ["DetectBase"]


class DetectBase(DetectGUI):
    def __init__(self) -> None:
        super(DetectBase, self).__init__()

        self.timer: QTimer = QTimer(self)
        self.timer.timeout.connect(self.on_timeout)

        self.results_queue: Queue[tuple[float, float]] = Queue()
        self.state_queue: Queue[tuple[int, int, int]] = Queue()
        self.good_to_measure: SharedMemory = SharedMemory(create=True, size=128)
        self.good_to_measure.buf[0] = False
        self.measurement: DetectMeasurement | None = None

        self.config: ConfigParser = ConfigParser(
            allow_no_value=True,
            inline_comment_prefixes=("#", ";"),
        )
        self.config.read("config.ini")

        print("connecting Triton...", end="", flush=True)
        self.triton: Triton = Triton(None, 33576)
        print(" done")
        self.triton.query_temperature(6, blocking=True)

        print("connecting APUASYN20...", end="", flush=True)
        self.synthesizer: APUASYN20 = APUASYN20()
        print(" done\n")

        self.sample_name: Final[str] = self.config.get("circuitry", "sample name")
        self.parameters_box.setTitle(self.sample_name)
        self.gain: Final[float] = get_float(
            self.config, self.sample_name, "circuitry", "voltage gain"
        )
        self.divider: Final[float] = get_float(
            self.config, self.sample_name, "circuitry", "current divider"
        )
        self.r: Final[float] = get_float(
            self.config,
            self.sample_name,
            "circuitry",
            "ballast resistance [Ohm]",
        ) + get_float(
            self.config,
            self.sample_name,
            "circuitry",
            "additional ballast resistance [Ohm]",
            fallback=0.0,
        )
        self.r_series: Final[float] = get_float(
            self.config,
            self.sample_name,
            "circuitry",
            "resistance in series [Ohm]",
            fallback=0.0,
        )

        self.setting_function: Final[str] = get_str(
            self.config, "current", self.sample_name, "function", fallback="sine"
        )
        if self.setting_function.casefold() not in ("linear", "sine"):
            raise ValueError(
                "Unsupported current setting function:", self.setting_function
            )

        self.bias_current_values: SliceSequence = SliceSequence(
            get_str(self.config, self.sample_name, "current", "bias current [nA]")
        )
        self.stop_key_bias.setDisabled(len(self.bias_current_values) <= 1)
        self.initial_biases: list[float] = get_float_list(
            self.config, self.sample_name, "current", "initial current [nA]", [0.0]
        )

        self.setting_time_values: Final[SliceSequence] = SliceSequence(
            get_str(self.config, self.sample_name, "current", "setting time [sec]")
        )
        self.stop_key_setting_time.setDisabled(len(self.setting_time_values) <= 1)

        self.check_exists: Final[bool] = self.config.getboolean(
            "measurement", "check whether file exists"
        )
        self.trigger_voltage: float = (
            get_float(
                self.config, self.sample_name, "measurement", "voltage trigger [V]"
            )
            * self.gain
        )
        self.max_reasonable_bias_error: Final[float] = (
            abs(
                self.config.getfloat(
                    "detect", "maximal reasonable bias error [%]", fallback=np.inf
                )
            )
            * 0.01
        )
        self.cycles_count: Final[int] = self.config.getint("detect", "number of cycles")
        self.max_switching_events_count: Final[int] = self.config.getint(
            "detect", "number of switches"
        )
        self.minimal_probability_to_measure: Final[float] = get_float(
            self.config,
            self.sample_name,
            "detect",
            "minimal probability to measure [%]",
            fallback=0.0,
        )
        self.delay_between_cycles_values: Final[SliceSequence] = SliceSequence(
            get_str(
                self.config,
                self.sample_name,
                "measurement",
                "delay between cycles [sec]",
            )
        )
        self.stop_key_delay_between_cycles.setDisabled(
            len(self.delay_between_cycles_values) <= 1
        )
        self.adc_rate: Final[float] = get_float(
            self.config,
            self.sample_name,
            "measurement",
            "adc rate [S/sec]",
            fallback=np.nan,
        )

        self.frequency_values: SliceSequence = SliceSequence(
            self.config.get("GHz signal", "frequency [GHz]")
        )
        self.stop_key_frequency.setDisabled(len(self.frequency_values) <= 1)
        self.power_dbm_values: SliceSequence = SliceSequence(
            self.config.get("GHz signal", "power [dBm]")
        )
        self.stop_key_power.setDisabled(len(self.power_dbm_values) <= 1)
        self.pulse_duration: Final[float] = get_float(
            self.config, self.sample_name, "detect", "GHz pulse duration [sec]"
        )
        self.waiting_after_pulse: Final[float] = get_float(
            self.config, self.sample_name, "detect", "waiting after GHz pulse [sec]"
        )

        self.temperature_values: SliceSequence = SliceSequence(
            self.config.get("measurement", "temperature")
        )
        self.temperature_delay: timedelta = timedelta(
            seconds=get_float(
                self.config,
                self.sample_name,
                "measurement",
                "time to wait for temperature [minutes]",
                fallback=0.0,
            )
            * 60.0
        )
        self.stop_key_temperature.setDisabled(len(self.temperature_values) <= 1)
        self.temperature_tolerance: Final[float] = (
            abs(
                get_float(
                    self.config,
                    self.sample_name,
                    "measurement",
                    "temperature tolerance [%]",
                    fallback=0.5,
                )
            )
            * 0.01
        )
        self.change_filtered_readings: Final[bool] = self.config.getboolean(
            "measurement", "change filtered readings in Triton", fallback=True
        )

        self.saving_location: Path = Path(
            self.config.get("output", "location", fallback=r"d:\ttt\detect")
        )
        self.saving_location /= self.sample_name
        self.saving_location /= date.today().isoformat()
        self.saving_location.mkdir(parents=True, exist_ok=True)

        self.temperature_index: int = 0
        self.frequency_index: int = 0
        self.setting_time_index: int = 0
        self.delay_between_cycles_index: int = 0
        self.bias_current_index: int = 0
        self.power_index: int = 0

        self.saved_files: set[Path] = set()

        self.bad_temperature_time: datetime = datetime.now() - self.temperature_delay
        self.temperature_just_set: bool = False

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
    def stat_file(self) -> Path:
        ...

    @property
    def data_file(self) -> Path:
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
                        f"CC{self.cycles_count}",
                        format_float(self.frequency, suffix="GHz"),
                        format_float(self.power_dbm, suffix="dBm"),
                        format_float(self.pulse_duration, prefix="P", suffix="s"),
                        format_float(
                            self.waiting_after_pulse, prefix="WaP", suffix="s"
                        ),
                        format_float(self.setting_time, prefix="ST", suffix="s"),
                        self.config.get("output", "suffix", fallback=""),
                    ),
                )
            )
            + ".txt"
        )

    @property
    @abc.abstractmethod
    def _line_index(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def _line_name(self) -> str:
        ...

    @property
    def plot_line(self) -> pg.PlotDataItem:
        i: int = self._line_index
        if i not in self.plot_lines:
            color: QColor = pg.intColor(i)
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

        self.synthesizer.pulse_modulation.source = "ext"
        self.synthesizer.pulse_modulation.state = True
        self.synthesizer.output = True

        self.measurement = DetectMeasurement(
            results_queue=self.results_queue,
            state_queue=self.state_queue,
            good_to_go=self.good_to_measure,
            resistance=self.r,
            resistance_in_series=self.r_series,
            current_divider=self.divider,
            current_setting_function=self.setting_function,
            initial_biases=self.initial_biases,
            cycles_count=self.cycles_count,
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

        self.triton.issue_temperature(6, self.temperature)
        self.label_temperature.setValue(self.temperature * 1000)
        self.label_setting_time.setValue(self.setting_time * 1000)
        self.label_delay_between_cycles.setValue(self.delay_between_cycles * 1000)
        self.synthesizer.frequency = self.frequency * 1e9
        self.label_frequency.setValue(self.frequency)
        self.label_bias.setValue(self.bias_current)
        self.synthesizer.power.level = self.power_dbm
        self.label_power.setValue(self.power_dbm)
        self.label_pulse_duration.setValue(self.pulse_duration * 1000)

        actual_temperature: float
        temperature_unit: str
        actual_temperature, temperature_unit = self.triton.query_temperature(6)
        self.temperature_just_set = not (
            (1.0 - self.temperature_tolerance) * self.temperature
            < actual_temperature
            < (1.0 + self.temperature_tolerance) * self.temperature
        )

        print(f"saving to {self.stat_file}")
        self.setWindowTitle(f"Detect — {self.stat_file}")
        self.timer.start(50)

    @abc.abstractmethod
    def _next_indices(self, make_step: bool = True) -> bool:
        ...

    def on_button_start_clicked(self) -> None:
        super(DetectBase, self).on_button_start_clicked()

        if self.check_exists and not self._next_indices(make_step=False):
            error("nothing left to measure")
            self.on_button_stop_clicked()
            return

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
        self.synthesizer.pulse_modulation.state = False
        self.synthesizer.output = False
        self.saved_files.add(self.data_file)
        super(DetectBase, self).on_button_stop_clicked()

    def _read_state_queue(self) -> None:
        cycle_index: int
        estimated_cycles_count: int
        switches_count: int
        while not self.state_queue.empty():
            cycle_index, estimated_cycles_count, switches_count = self.state_queue.get(
                block=True
            )
            self.label_loop_number.setValue(cycle_index + 1)
            self.label_loop_count.setValue(estimated_cycles_count)
            self.label_probability.setValue(switches_count / (cycle_index + 1) * 100)

    def _add_plot_point(self, x: float, prob: float, err: float) -> None:
        old_x_data: NDArray[np.float64] = (
            np.empty(0, dtype=np.float64)
            if self.plot_line.xData is None
            else self.plot_line.xData
        )
        old_y_data: NDArray[np.float64] = (
            np.empty(0, dtype=np.float64)
            if self.plot_line.yData is None
            else self.plot_line.yData
        )
        x_data: NDArray[np.float64] = np.append(old_x_data, x)
        y_data: NDArray[np.float64] = np.append(old_y_data, prob)
        self.plot_line.setData(x_data, y_data)

    def _watch_temperature(self) -> None:
        td: timedelta
        actual_temperature: float
        temperature_unit: str
        actual_temperature, temperature_unit = self.triton.query_temperature(6)
        ats: bytes = str(actual_temperature * 1000).encode()
        self.good_to_measure.buf[1 : 1 + len(ats)] = ats
        if not (
            (1.0 - self.temperature_tolerance) * self.temperature
            < actual_temperature
            < (1.0 + self.temperature_tolerance) * self.temperature
        ):
            self.good_to_measure.buf[0] = False
            self.bad_temperature_time = datetime.now()
            self.timer.setInterval(1000)
            print(
                f"temperature {actual_temperature} {temperature_unit} "
                f"is too far from {self.temperature:.3f} K"
            )
            if not self.triton.issue_temperature(6, self.temperature):
                error(f"failed to set temperature to {self.temperature} K")
                self.timer.stop()
                self.measurement.terminate()
            if self.change_filtered_readings:
                if not self.triton.issue_filter_readings(
                    6, self.triton.filter_readings(self.temperature)
                ):
                    error("failed to change the state of filtered readings")
                    self.timer.stop()
                    self.measurement.terminate()
            if not self.triton.issue_heater_range(
                6, self.triton.heater_range(self.temperature)
            ):
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
                    f"temperature {actual_temperature} {temperature_unit} "
                    f"is close enough to {self.temperature:.3f} K, but not for long enough yet"
                    f": {self.temperature_delay - td} left"
                )
                self.timer.setInterval(1000)
        else:
            self.good_to_measure.buf[0] = True

    @abc.abstractmethod
    def _add_plot_point_from_file(self) -> None:
        ...

    def _data_file_exists(self, verbose: bool = True) -> bool:
        exists: bool = (
            self.bias_current_index < len(self.bias_current_values)
            and self.power_index < len(self.power_dbm_values)
            and self.frequency_index < len(self.frequency_values)
            and self.setting_time_index < len(self.setting_time_values)
            and self.delay_between_cycles_index < len(self.delay_between_cycles_values)
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
    def on_timeout(self) -> None:
        ...
