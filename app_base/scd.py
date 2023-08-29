# -*- coding: utf-8 -*-
from __future__ import annotations

import abc
from configparser import ConfigParser
from datetime import date, datetime, timedelta
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Final, TextIO, cast

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from qtpy.QtCore import QTimer, Qt
from qtpy.QtGui import QCloseEvent, QColor

from backend.communication.anapico_communication import APUASYN20
from backend.communication.triton_communication import Triton
from backend.measurement.scd import SCDMeasurement
from backend.utils import SliceSequence, error, warning
from backend.utils.config import *
from backend.utils.string_utils import format_float
from ui.scd_gui import SwitchingCurrentDistributionGUI

__all__ = ['SwitchingCurrentDistributionBase']


class SwitchingCurrentDistributionBase(SwitchingCurrentDistributionGUI):
    def __init__(self, flags=Qt.WindowFlags()) -> None:
        super().__init__(flags=flags)

        self.timer: QTimer = QTimer(self)
        self.timer.timeout.connect(self.on_timeout)

        self.results_queue: Queue[tuple[float, float]] = Queue()
        self.state_queue: Queue[tuple[int, timedelta]] = Queue()
        self.switching_data_queue: Queue[tuple[np.float64, np.float64]] = Queue()
        self.switching_current: list[np.float64] = []
        self.switching_voltage: list[np.float64] = []
        self.good_to_measure: SharedMemory = SharedMemory(create=True, size=128)
        self.good_to_measure.buf[0] = False
        self.measurement: SCDMeasurement | None = None

        self.config: ConfigParser = ConfigParser(allow_no_value=True, inline_comment_prefixes=('#', ';'))
        self.config.read('config.ini')

        self.triton: Triton = Triton(None, 33576)
        self.triton.query_temperature(6, blocking=True)

        self.synthesizer: APUASYN20 = APUASYN20(expected=self.config.getboolean('GHz signal', 'connect', fallback=True))

        self.sample_name: Final[str] = self.config.get('circuitry', 'sample name')
        self.parameters_box.setTitle(self.sample_name)
        self.gain: Final[float] = get_float(self.config, self.sample_name, 'circuitry', 'voltage gain')
        self.divider: Final[float] = get_float(self.config, self.sample_name, 'circuitry', 'current divider')
        self.r: Final[float] = (get_float(self.config, self.sample_name, 'circuitry', 'ballast resistance [Ohm]')
                                + get_float(self.config, self.sample_name, 'circuitry',
                                            'additional ballast resistance [Ohm]', fallback=0.0))
        self.r_series: Final[float] = get_float(self.config, self.sample_name, 'circuitry',
                                                'resistance in series [Ohm]', fallback=0.0)

        self.reset_function: Final[str] = get_str(self.config, 'current', self.sample_name, 'function', fallback='sine')
        if self.reset_function.casefold() not in ('linear', 'sine'):
            raise ValueError('Unsupported current reset function:', self.reset_function)
        self.max_bias_current: float = get_float(self.config, self.sample_name, 'scd', 'max bias current [nA]')
        self.initial_biases: list[float] = get_float_list(self.config, self.sample_name,
                                                          'current', 'initial current [nA]', [0.0])
        self.current_speed_values: Final[SliceSequence] = \
            SliceSequence(get_str(self.config, self.sample_name, 'scd', 'current speed [nA/sec]'))
        self.stop_key_current_speed.setDisabled(len(self.current_speed_values) <= 1)

        self.check_exists: Final[bool] = self.config.getboolean('measurement', 'check whether file exists')
        self.trigger_voltage: float = get_float(self.config, self.sample_name, 'measurement', 'voltage trigger [V]')
        self.max_reasonable_bias_error: Final[float] = abs(self.config.getfloat('scd',
                                                                                'maximal reasonable bias error [%]',
                                                                                fallback=np.inf))
        self.cycles_count: int = self.config.getint('scd', 'number of cycles')
        self.max_measurement_time: timedelta = \
            timedelta(seconds=self.config.getfloat('scd', 'max cycles measurement time [minutes]') * 60)
        self.delay_between_cycles_values: Final[SliceSequence] \
            = SliceSequence(get_str(self.config, self.sample_name, 'measurement', 'delay between cycles [sec]'))
        self.stop_key_delay_between_cycles.setDisabled(len(self.delay_between_cycles_values) <= 1)
        self.adc_rate: Final[float] = get_float(self.config, self.sample_name,
                                                'measurement', 'adc rate [S/sec]',
                                                fallback=np.nan)

        synthesizer_output: bool = self.config.getboolean('GHz signal', 'on', fallback=False)
        self.frequency_values: SliceSequence = SliceSequence(self.config.get('GHz signal', 'frequency [GHz]'))
        self.stop_key_frequency.setDisabled(not synthesizer_output or len(self.frequency_values) <= 1)
        self.power_dbm_values: SliceSequence = SliceSequence(self.config.get('GHz signal', 'power [dBm]'))
        self.stop_key_power.setDisabled(not synthesizer_output or len(self.power_dbm_values) <= 1)

        self.temperature_values: SliceSequence = SliceSequence(self.config.get('measurement', 'temperature'))
        self.temperature_delay: timedelta = \
            timedelta(seconds=self.config.getfloat('measurement', 'time to wait for temperature [minutes]',
                                                   fallback=0.0) * 60.)
        self.change_filtered_readings: Final[bool] = self.config.getboolean('measurement',
                                                                            'change filtered readings in Triton',
                                                                            fallback=True)
        self.stop_key_temperature.setDisabled(len(self.temperature_values) <= 1)
        self.temperature_tolerance: Final[float] = abs(self.config.getfloat('measurement', 'temperature tolerance [%]',
                                                                            fallback=1.0))

        self.saving_location: Path = Path(self.config.get('output', 'location', fallback=r'd:\ttt\scd'))
        self.saving_location /= self.sample_name
        self.saving_location /= date.today().isoformat()
        self.saving_location.mkdir(parents=True, exist_ok=True)

        self.temperature_index: int = 0
        self.current_speed_index: int = 0
        self.frequency_index: int = 0
        self.delay_between_cycles_index: int = 0
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
    def power_dbm(self) -> float:
        return float(self.power_dbm_values[self.power_index]) if self.synthesizer_output else np.nan

    @property
    def frequency(self) -> float:
        return float(self.frequency_values[self.frequency_index]) if self.synthesizer_output else np.nan

    @property
    def current_speed(self) -> float:
        return self.current_speed_values[self.current_speed_index]

    @property
    def delay_between_cycles(self) -> float:
        return self.delay_between_cycles_values[self.delay_between_cycles_index]

    @property
    @abc.abstractmethod
    def stat_file(self) -> Path:
        ...

    @property
    def data_file(self) -> Path:
        return self.saving_location / (' '.join(filter(None, (
            'I''SCD',
            self.config.get('output', 'prefix', fallback=''),
            format_float(self.temperature * 1e3, suffix='mK'),
            format_float(self.current_speed, prefix='v', suffix='nAps'),
            format_float(self.delay_between_cycles, prefix='d', suffix='s'),
            f'CC{self.cycles_count}',
            format_float(self.frequency, suffix='GHz')
            if self.synthesizer_output else '',
            format_float(self.power_dbm, suffix='dBm')
            if self.synthesizer_output else '',
            format_float(self.initial_biases[-1], prefix='from ', suffix='nA'),
            format_float(self.trigger_voltage * 1e3, prefix='threshold', suffix='mV'),
            self.config.get('output', 'suffix', fallback='')
        ))) + '.txt')

    @property
    def hist_file(self) -> Path:
        return self.saving_location / (' '.join(filter(None, (
            'I''SCD-hist',
            self.config.get('output', 'prefix', fallback=''),
            format_float(self.temperature * 1e3, suffix='mK'),
            format_float(self.current_speed, prefix='v', suffix='nAps'),
            format_float(self.delay_between_cycles, prefix='d', suffix='s'),
            f'CC{self.cycles_count}',
            format_float(self.frequency, suffix='GHz')
            if self.synthesizer_output else '',
            format_float(self.power_dbm, suffix='dBm')
            if self.synthesizer_output else '',
            format_float(self.initial_biases[-1], prefix='from ', suffix='nA'),
            format_float(self.trigger_voltage * 1e3, prefix='threshold', suffix='mV'),
            self.config.get('output', 'suffix', fallback='')
        ))) + '.txt')

    @property
    @abc.abstractmethod
    def _line_index(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def _line_name(self) -> str:
        ...

    @property
    def plot_line_mean(self) -> pg.PlotDataItem:
        i: int = self._line_index
        if i not in self.plot_lines_mean:
            color: QColor = pg.intColor(i)
            self.plot_lines_mean[i] = self.canvas_mean.plot(np.empty(0), symbol='o',
                                                            name=self._line_name or None,
                                                            pen=color, symbolPen=color, symbolBrush=color)
        return self.plot_lines_mean[i]

    @property
    def plot_line_std(self) -> pg.PlotDataItem:
        i: int = self._line_index
        if i not in self.plot_lines_std:
            color: QColor = pg.intColor(i)
            self.plot_lines_std[i] = self.canvas_std.plot(np.empty(0), symbol='o',
                                                          name=self._line_name or None,
                                                          pen=color, symbolPen=color, symbolBrush=color)
        return self.plot_lines_std[i]

    def start_measurement(self) -> None:
        if self.measurement is not None and self.measurement.is_alive():
            self.measurement.terminate()
            self.measurement.join()

        self.switching_current = []
        self.switching_voltage = []
        self.synthesizer.output = self.synthesizer_output
        self.synthesizer.power.alc.low_noise = True

        self.measurement = SCDMeasurement(results_queue=self.results_queue, state_queue=self.state_queue,
                                          switching_data_queue=self.switching_data_queue,
                                          good_to_go=self.good_to_measure,
                                          resistance=self.r,
                                          resistance_in_series=self.r_series,
                                          current_divider=self.divider,
                                          current_reset_function=self.reset_function,
                                          initial_biases=self.initial_biases,
                                          cycles_count=self.cycles_count,
                                          max_bias_current=self.max_bias_current,
                                          power_dbm=self.power_dbm,
                                          current_speed=self.current_speed,
                                          trigger_voltage=self.trigger_voltage,
                                          voltage_gain=self.gain,
                                          temperature=self.temperature,
                                          stat_file=self.stat_file,
                                          frequency=self.frequency,
                                          data_file=self.data_file,
                                          max_measurement_time=self.max_measurement_time,
                                          max_reasonable_bias_error=self.max_reasonable_bias_error,
                                          delay_between_cycles=self.delay_between_cycles,
                                          adc_rate=self.adc_rate)
        self.measurement.start()

        self.triton.issue_temperature(6, self.temperature)
        self.label_temperature.setValue(self.temperature * 1000)
        self.label_current_speed.setValue(self.current_speed)
        self.label_delay_between_cycles.setValue(self.delay_between_cycles * 1000)
        self.synthesizer.frequency = self.frequency * 1e9
        self.label_frequency.setValue(self.frequency)
        self.synthesizer.power.level = self.power_dbm
        self.label_power.setValue(self.power_dbm)

        actual_temperature: float
        temperature_unit: str
        actual_temperature, temperature_unit = self.triton.query_temperature(6)
        self.temperature_just_set = not (
                (1.0 - 0.01 * self.temperature_tolerance) * self.temperature
                < actual_temperature
                < (1.0 + 0.01 * self.temperature_tolerance) * self.temperature)

        print(f'\nsaving to {self.stat_file}')
        self.setWindowTitle(f'Switching Current Distribution — {self.stat_file}')
        self.timer.start(50)

    @property
    def synthesizer_output(self) -> bool:
        return self.config.getboolean('GHz signal', 'on', fallback=False)

    @abc.abstractmethod
    def _next_indices(self, make_step: bool = True) -> bool:
        ...

    def on_button_start_clicked(self) -> None:
        super(SwitchingCurrentDistributionBase, self).on_button_start_clicked()

        if self.check_exists and not self._next_indices(make_step=False):
            error('nothing left to measure')
            self.on_button_stop_clicked()
            return

        if self.stat_file.exists():
            f_out: TextIO
            with self.stat_file.open('at', encoding='utf-8') as f_out:
                f_out.write('\n')
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
        super(SwitchingCurrentDistributionBase, self).on_button_stop_clicked()

    def _read_state_queue(self) -> None:
        cycle_index: int
        remaining_time: timedelta
        while not self.state_queue.empty():
            cycle_index, remaining_time = self.state_queue.get(block=True)
            self.label_loop_number.setValue(cycle_index)
            self.label_remaining_time.setText(str(remaining_time)[:9])

    def _read_switching_data_queue(self) -> None:
        current: np.float64
        voltage: np.float64
        while not self.switching_data_queue.empty():
            current, voltage = self.switching_data_queue.get(block=True)
            self.switching_current.append(current)
            self.switching_voltage.append(voltage)
            self.label_mean_current.setValue(np.nanmean(self.switching_current) * 1e9)
            self.label_std_current.setValue(np.nanstd(self.switching_current) * 1e9)
            self.histogram.hist(self.switching_current, pen='white', symbolBrush='white', symbolPen='white')

    def _add_plot_point(self, x: float, mean: float, std: float) -> None:
        old_x_data: NDArray[np.float64] = (np.empty(0)
                                           if self.plot_line_mean.xData is None
                                           else self.plot_line_mean.xData)
        old_mean_data: NDArray[np.float64] = (np.empty(0)
                                              if self.plot_line_mean.yData is None
                                              else self.plot_line_mean.yData)
        old_std_data: NDArray[np.float64] = (np.empty(0)
                                             if self.plot_line_std.yData is None
                                             else self.plot_line_std.yData)
        x_data: NDArray[np.float64] = np.append(old_x_data, x)
        mean_data: NDArray[np.float64] = np.append(old_mean_data, mean)
        std_data: NDArray[np.float64] = np.append(old_std_data, std)
        self.plot_line_mean.setData(x_data, mean_data)
        self.plot_line_std.setData(x_data, std_data)

    def _add_plot_point_from_file(self, x: float) -> None:
        if self.data_file in self.saved_files:
            return
        self.saved_files.add(self.data_file)
        measured_data: NDArray[float] = self._get_data_file_content()
        if measured_data.shape[0] == 3:
            current: NDArray[float] = measured_data[0] * 1e9
            median_bias_current: float = cast(float, np.nanmedian(current))
            min_reasonable_bias_current: float = median_bias_current * (1. - .01 * self.max_reasonable_bias_error)
            max_reasonable_bias_current: float = median_bias_current * (1. + .01 * self.max_reasonable_bias_error)
            reasonable: NDArray[np.bool_] = ((current >= min_reasonable_bias_current)
                                             & (current <= max_reasonable_bias_current))
            current = current[reasonable]
            self._add_plot_point(x,
                                 cast(float, np.mean(current)),
                                 cast(float, np.std(current)))

    def _watch_temperature(self) -> None:
        td: timedelta
        actual_temperature: float
        temperature_unit: str
        actual_temperature, temperature_unit = self.triton.query_temperature(6)
        ats: bytes = str(actual_temperature * 1000).encode()
        self.good_to_measure.buf[1:1 + len(ats)] = ats
        if not ((1.0 - 0.01 * self.temperature_tolerance) * self.temperature
                < actual_temperature
                < (1.0 + 0.01 * self.temperature_tolerance) * self.temperature):
            self.good_to_measure.buf[0] = False
            self.bad_temperature_time = datetime.now()
            self.timer.setInterval(1000)
            print(f'temperature {actual_temperature} {temperature_unit} '
                  f'is too far from {self.temperature:.3f} K')
            if not self.triton.issue_temperature(6, self.temperature):
                error(f'failed to set temperature to {self.temperature} K')
                self.timer.stop()
                self.measurement.terminate()
            if self.change_filtered_readings:
                if not self.triton.issue_filter_readings(6, self.triton.filter_readings(self.temperature)):
                    error(f'failed to change the state of filtered readings')
                    self.timer.stop()
                    self.measurement.terminate()
            if not self.triton.issue_heater_range(6, self.triton.heater_range(self.temperature)):
                error(f'failed to change the heater range')
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
                print(f'temperature {actual_temperature} {temperature_unit} '
                      f'is close enough to {self.temperature:.3f} K, but not for long enough yet'
                      f': {self.temperature_delay - td} left')
                self.timer.setInterval(1000)
        else:
            self.good_to_measure.buf[0] = True

    def _data_file_exists(self, verbose: bool = True) -> bool:
        exists: bool = (self.power_index < len(self.power_dbm_values)
                        and self.frequency_index < len(self.frequency_values)
                        and self.current_speed_index < len(self.current_speed_values)
                        and self.delay_between_cycles_index < len(self.delay_between_cycles_values)
                        and self.temperature_index < len(self.temperature_values)
                        and self.data_file.exists()
                        and self._get_data_file_content().size)
        if exists and verbose:
            if self.data_file not in self.saved_files:
                warning(f'{self.data_file} already exists')
        return exists

    def _get_data_file_content(self) -> NDArray[float]:
        return np.array([[float(cell) for cell in row.split('\t')]
                         for row in self.data_file.read_text(encoding='utf-8').splitlines()
                         if row and (row.startswith('nan') or not row[0].isalpha())]).T

    @abc.abstractmethod
    def on_timeout(self) -> None:
        ...
