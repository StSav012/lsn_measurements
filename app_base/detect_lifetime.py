# -*- coding: utf-8 -*-
from __future__ import annotations

import abc
from configparser import ConfigParser
from datetime import date, datetime, timedelta
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Final, Literal, Optional, TextIO

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QCloseEvent, QColor

from backend.communication.anapico_communication import APUASYN20
from backend.communication.triton_communication import Triton
from backend.measurement.detect import DetectMeasurement
from backend.measurement.lifetime import LifetimeMeasurement
from backend.utils import SliceSequence, error, warning
from backend.utils.config import *
from ui.detect_lifetime_gui import DetectLifetimeGUI

__all__ = ['DetectLifetimeBase']


class DetectLifetimeBase(DetectLifetimeGUI):
    def __init__(self, flags=Qt.WindowFlags()) -> None:
        super(DetectLifetimeBase, self).__init__(flags=flags)

        self.timer: QTimer = QTimer(self)
        self.timer.timeout.connect(self.on_timeout)

        self.results_queue_detect: Queue[tuple[float, float]] = Queue()
        self.state_queue_detect: Queue[tuple[int, int, int]] = Queue()
        self.results_queue_lifetime: Queue[tuple[float, float, float]] = Queue()
        self.state_queue_lifetime: Queue[tuple[int, timedelta]] = Queue()
        self.good_to_measure: SharedMemory = SharedMemory(create=True, size=1)
        self.good_to_measure.buf[0] = False
        self.measurement: Optional[DetectMeasurement | LifetimeMeasurement] = None

        self.config: ConfigParser = ConfigParser(allow_no_value=True, inline_comment_prefixes=('#', ';'))
        self.config.read('config.ini')

        self.triton: Triton = Triton('192.168.199.89', 33576)
        self.triton.query_temperature(6, blocking=True)

        self.synthesizer: APUASYN20 = APUASYN20('192.168.199.109',
                                                expected=self.config.getboolean('GHz signal', 'connect', fallback=True))

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
        self.bias_current_values: SliceSequence = SliceSequence(get_str(self.config, self.sample_name,
                                                                        'current', 'bias current [nA]'))
        self.stop_key_bias.setDisabled(len(self.bias_current_values) <= 1)
        self.initial_biases: list[float] = list(map(float, get_str(self.config, self.sample_name, 'current',
                                                                   'initial current [nA]').split(',')))
        self.setting_time: Final[float] = get_float(self.config, self.sample_name, 'current', 'setting time [sec]')

        self.check_exists: Final[bool] = self.config.getboolean('measurement', 'check whether file exists')
        self.trigger_voltage: float = get_float(self.config, self.sample_name,
                                                'measurement', 'voltage trigger [V]') * self.gain
        self.max_reasonable_bias_error: Final[float] = abs(self.config.getfloat('lifetime',
                                                                                'maximal reasonable bias error [%]',
                                                                                fallback=np.inf))
        self.cycles_count_lifetime: int = self.config.getint('lifetime', 'number of cycles')
        self.cycles_count_detect: Final[int] = self.config.getint('detect', 'number of cycles')
        self.max_switching_events_count: Final[int] = self.config.getint('detect', 'number of switches')
        self.minimal_probability_to_measure: Final[float] \
            = self.config.getfloat('detect', 'minimal probability to measure [%]', fallback=0.0)
        self.max_waiting_time: timedelta = \
            timedelta(seconds=self.config.getfloat('lifetime', 'max time of waiting for switching [sec]'))
        self.max_mean: Final[float] = self.config.getfloat('lifetime', 'max mean time to measure [sec]',
                                                           fallback=np.inf)
        self.ignore_never_switched: bool = self.config.getboolean('lifetime', 'ignore never switched')
        self.delay_between_cycles: Final[float] = get_float(self.config, self.sample_name,
                                                            'measurement', 'delay between cycles [sec]',
                                                            fallback=0.0)

        self.frequency_values: SliceSequence = SliceSequence(self.config.get('GHz signal', 'frequency [GHz]'))
        self.stop_key_frequency.setDisabled(len(self.frequency_values) <= 1)
        self.power_dbm_values: SliceSequence = SliceSequence(self.config.get('GHz signal', 'power [dBm]'))
        self.stop_key_power.setDisabled(len(self.power_dbm_values) <= 1)
        self.pulse_duration: Final[float] = self.config.getfloat('detect', 'GHz pulse duration [sec]')
        self.waiting_after_pulse: Final[float] = self.config.getfloat('detect', 'waiting after GHz pulse [sec]')

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

        self.saving_location: Path = Path(self.config.get('output', 'location', fallback=r'd:\ttt\detect+lifetime'))
        self.saving_location /= self.sample_name
        self.saving_location /= date.today().isoformat()
        self.saving_location.mkdir(parents=True, exist_ok=True)

        self.temperature_index: int = 0
        self.frequency_index: int = 0
        self.bias_current_index: int = 0
        self.power_index: int = 0

        self.last_lifetime_0: float = np.nan
        self.bad_temperature_time: datetime = datetime.now() - self.temperature_delay
        self.temperature_just_set: bool = False

        self.mode: Literal['detect', 'lifetime'] = 'detect'

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
    @abc.abstractmethod
    def stat_file(self) -> Path:
        ...

    @property
    def data_file_lifetime(self) -> Path:
        return self.saving_location / (' '.join((
            'lifetimes',
            self.config.get('output', 'prefix', fallback=''),
            f'{self.temperature * 1e3:.6f}'.rstrip('0').rstrip('.') + 'mK',
            f'{self.bias_current:.6f}'.rstrip('0').rstrip('.') + 'nA',
            f'd{self.delay_between_cycles:.6f}'.rstrip('0').rstrip('.') + 's',
            f'CC{self.cycles_count_lifetime}',
            f'ST{self.setting_time:.6f}'.rstrip('0').rstrip('.') + 's',
            f'{self.frequency:.6f}'.rstrip('0').rstrip('.') + 'GHz'
            if not np.isnan(self.frequency) else '',
            f'{self.power_dbm:.6f}'.rstrip('0').rstrip('.') + 'dBm'
            if not np.isnan(self.power_dbm) else '',
            f'from {self.initial_biases[-1]:.6f}'.rstrip('0').rstrip('.') + 'nA',
            self.config.get('output', 'suffix', fallback='')
        )).replace('  ', ' ').replace('  ', ' ').strip(' ') + '.txt')

    @property
    @abc.abstractmethod
    def _line_index_detect(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def _line_name_detect(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def _line_index_lifetime(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def _line_name_lifetime(self) -> str:
        ...

    @property
    def plot_line_detect(self) -> pg.PlotDataItem:
        i: int = self._line_index_detect
        if i not in self.plot_lines_detect:
            color: QColor = pg.intColor(i)
            self.plot_lines_detect[i] = self.canvas_detect.plot(np.empty(0), symbol='o',
                                                                name=self._line_name_detect,
                                                                pen=color, symbolPen=color, symbolBrush=color)
        return self.plot_lines_detect[i]

    @property
    def plot_line_lifetime(self) -> pg.PlotDataItem:
        i: int = self._line_index_lifetime
        if i not in self.plot_lines_lifetime:
            color: QColor = pg.intColor(i)
            self.plot_lines_lifetime[i] = self.canvas_lifetime.plot(np.empty(0), symbol='o',
                                                                    name=self._line_name_lifetime,
                                                                    pen=color, symbolPen=color, symbolBrush=color)
        return self.plot_lines_lifetime[i]

    def start_measurement_detect(self) -> None:
        if self.measurement is not None and self.measurement.is_alive():
            self.measurement.terminate()
            self.measurement.join()
        self.synthesizer.output = self.mode == 'detect'
        self.measurement = DetectMeasurement(results_queue=self.results_queue_detect,
                                             state_queue=self.state_queue_detect,
                                             good_to_go=self.good_to_measure,
                                             resistance=self.r,
                                             resistance_in_series=self.r_series,
                                             current_divider=self.divider,
                                             current_setting_function=self.reset_function,
                                             initial_biases=self.initial_biases,
                                             cycles_count=self.cycles_count_detect,
                                             bias_current=self.bias_current,
                                             power_dbm=self.power_dbm,
                                             max_switching_events_count=self.max_switching_events_count,
                                             pulse_duration=self.pulse_duration,
                                             setting_time=self.setting_time,
                                             trigger_voltage=self.trigger_voltage,
                                             voltage_gain=self.gain,
                                             temperature=self.temperature,
                                             stat_file=self.stat_file,
                                             frequency=self.frequency,
                                             waiting_after_pulse=self.waiting_after_pulse)
        self.measurement.start()
        self.temperature_just_set = False
        print(f'saving to {self.stat_file}')
        self.timer.start(50)

    def start_measurement_lifetime(self) -> None:
        if self.measurement is not None and self.measurement.is_alive():
            self.measurement.terminate()
            self.measurement.join()
        self.synthesizer.output = self.mode != 'lifetime'
        self.label_frequency.setValue(np.nan)
        self.label_power.setValue(np.nan)
        self.measurement = LifetimeMeasurement(results_queue=self.results_queue_lifetime,
                                               state_queue=self.state_queue_lifetime,
                                               good_to_go=self.good_to_measure,
                                               resistance=self.r,
                                               resistance_in_series=self.r_series,
                                               current_divider=self.divider,
                                               current_setting_function=self.reset_function,
                                               initial_biases=self.initial_biases,
                                               cycles_count=self.cycles_count_lifetime,
                                               bias_current=self.bias_current,
                                               power_dbm=self.power_dbm,
                                               setting_time=self.setting_time,
                                               trigger_voltage=self.trigger_voltage,
                                               voltage_gain=self.gain,
                                               temperature=self.temperature,
                                               stat_file=self.stat_file,
                                               frequency=self.frequency,
                                               data_file=self.data_file_lifetime,
                                               ignore_never_switched=self.ignore_never_switched,
                                               max_waiting_time=self.max_waiting_time,
                                               max_reasonable_bias_error=self.max_reasonable_bias_error,
                                               delay_between_cycles=self.delay_between_cycles)
        self.measurement.start()
        self.temperature_just_set = False
        print(f'saving to {self.stat_file}')
        self.timer.start(50)

    def start_measurement(self) -> None:
        {'detect': self.start_measurement_detect, 'lifetime': self.start_measurement_lifetime}[self.mode]()

    @abc.abstractmethod
    def _next_indices(self) -> bool:
        ...

    def on_button_start_clicked(self) -> None:
        super(DetectLifetimeBase, self).on_button_start_clicked()

        self.synthesizer.output = self.mode == 'detect'
        self.synthesizer.power.alc.low_noise = True

        if self.mode == 'detect':
            while self.check_exists and self.stat_file.exists():
                warning(f'{self.stat_file} already exists')
                if not self._next_indices():
                    error('nothing left to measure')
                    self.synthesizer.pulse_modulation.state = False
                    self.synthesizer.output = False
                    self.on_button_stop_clicked()
                    return
        elif self.mode == 'lifetime':
            if self.check_exists:
                while self.data_file_lifetime.exists():
                    warning(f'{self.data_file_lifetime} already exists')
                    if not self._next_indices():
                        error('nothing left to measure')
                        self.on_button_stop_clicked()
                        return

        self.triton.issue_temperature(6, self.temperature)
        self.label_temperature.setValue(self.temperature * 1000)
        self.synthesizer.frequency = self.frequency * 1e9
        self.label_frequency.setValue(self.frequency)
        self.label_bias.setValue(self.bias_current)
        self.synthesizer.power.level = self.power_dbm
        self.label_power.setValue(self.power_dbm)
        if self.stat_file.exists():
            f_out: TextIO
            with self.stat_file.open('at', encoding='utf-8') as f_out:
                f_out.write('\n')
        self.start_measurement()

    def on_button_stop_clicked(self) -> None:
        if self.measurement is not None:
            self.measurement.terminate()
            self.measurement.join()
        self.timer.stop()
        self.synthesizer.output = False
        super(DetectLifetimeBase, self).on_button_stop_clicked()

    @abc.abstractmethod
    def on_timeout(self) -> None:
        ...
