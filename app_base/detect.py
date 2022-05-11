# -*- coding: utf-8 -*-
import abc
import sys
from configparser import ConfigParser
from datetime import date, datetime, timedelta
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Dict, Final, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QCloseEvent, QColor
from PyQt5.QtWidgets import QApplication

from backend.communication.anapico_communication import APUASYN20
from backend.communication.triton_communication import Triton
from backend.measurement.detect import DetectMeasurement
from backend.utils import SliceSequence, error, warning, zero_sources
from backend.utils.config import *
from ui.detect_gui import DetectGUI

__all__ = ['DetectBase']


class DetectBase(abc.ABC, DetectGUI):
    def __init__(self, flags=Qt.WindowFlags()) -> None:
        super(DetectBase, self).__init__(flags=flags)

        self.timer: QTimer = QTimer(self)
        self.timer.timeout.connect(self.on_timeout)

        self.results_queue: Queue[Tuple[float, float]] = Queue()
        self.state_queue: Queue[Tuple[int, int, int]] = Queue()
        self.good_to_measure: SharedMemory = SharedMemory(create=True, size=1)
        self.good_to_measure.buf[0] = False
        self.measurement: Optional[DetectMeasurement] = None

        self.config: ConfigParser = ConfigParser(allow_no_value=True, inline_comment_prefixes=('#', ';'))
        self.config.read('config.ini')

        sys.stdout.write('connecting Triton...')
        self.triton: Triton = Triton('192.168.199.89', 33576)
        sys.stdout.write(' done\n')
        self.triton.query_temperature(6, blocking=True)

        sys.stdout.write('connecting APUASYN20...')
        self.synthesizer: APUASYN20 = APUASYN20('192.168.199.109')
        sys.stdout.write(' done\n')

        self.sample_name: Final[str] = self.config.get('circuitry', 'sample name')
        self.parameters_box.setTitle(self.sample_name)
        self.gain: Final[float] = get_float(self.config, self.sample_name, 'circuitry', 'voltage gain')
        self.divider: Final[float] = get_float(self.config, self.sample_name, 'circuitry', 'current divider')
        self.r: Final[float] = (get_float(self.config, self.sample_name, 'circuitry', 'ballast resistance [Ohm]')
                                + get_float(self.config, self.sample_name, 'circuitry',
                                            'additional ballast resistance [Ohm]', fallback=0.0))
        self.r_series: Final[float] = get_float(self.config, self.sample_name, 'circuitry',
                                                'resistance in series [Ohm]', fallback=0.0)

        self.setting_function: Final[str] = get_str(self.config, 'current', self.sample_name, 'function',
                                                    fallback='sine')
        self.bias_currents: SliceSequence = SliceSequence(get_str(self.config, self.sample_name,
                                                                  'current', 'bias current [nA]'))
        self.stop_key_bias.setDisabled(len(self.bias_currents) <= 1)
        self.initial_biases: Tuple[float] = get_float_tuple(self.config, self.sample_name,
                                                            'current', 'initial current [nA]')

        self.setting_time: Final[float] = get_float(self.config, self.sample_name, 'current', 'setting time [sec]')
        if self.setting_function.casefold() not in ('linear', 'sine'):
            raise ValueError('Unsupported current setting function:', self.setting_function)

        self.check_exists: Final[bool] = self.config.getboolean('measurement', 'check whether file exists')
        self.trigger_voltage: float = \
            get_float(self.config, self.sample_name, 'measurement', 'voltage trigger [V]') * self.gain
        self.cycles_count: Final[int] = self.config.getint('detect', 'number of cycles')
        self.max_switching_events_count: Final[int] = self.config.getint('detect', 'number of switches')
        self.minimal_probability_to_measure: Final[float] \
            = self.config.getfloat('detect', 'minimal probability to measure [%]', fallback=0.0)

        self.frequency_values: SliceSequence = SliceSequence(self.config.get('GHz signal', 'frequency [GHz]'))
        self.stop_key_frequency.setDisabled(len(self.frequency_values) <= 1)
        self.power_dbm_values: SliceSequence = SliceSequence(self.config.get('GHz signal', 'power [dBm]'))
        self.stop_key_power.setDisabled(len(self.power_dbm_values) <= 1)
        self.pulse_duration: Final[float] = self.config.getfloat('detect', 'GHz pulse duration [sec]')
        self.waiting_after_pulse: Final[float] = self.config.getfloat('detect', 'waiting after GHz pulse [sec]')

        self.saving_location: Path = Path(self.config.get('output', 'location', fallback=r'd:\ttt\detect'))
        self.saving_location /= self.sample_name
        self.saving_location /= date.today().isoformat()
        self.saving_location.mkdir(parents=True, exist_ok=True)

        self.temperature_values: SliceSequence = SliceSequence(self.config.get('measurement', 'temperature'))
        self.temperature_delay: timedelta = \
            timedelta(seconds=self.config.getfloat('measurement', 'time to wait for temperature [minutes]',
                                                   fallback=0.0) * 60.)
        self.stop_key_temperature.setDisabled(len(self.temperature_values) <= 1)
        self.temperature_tolerance: Final[float] = abs(self.config.getfloat('measurement', 'temperature tolerance [%]',
                                                                            fallback=0.5))
        self.change_filtered_readings: Final[bool] = self.config.getboolean('measurement',
                                                                            'change filtered readings in Triton',
                                                                            fallback=True)

        self.plot_lines: Dict[int, pg.PlotDataItem] = dict()
        self.figure.addLegend(offset=(-30, -30))

        self.temperature_index: int = 0
        self.frequency_index: int = 0
        self.bias_current_index: int = 0
        self.power_index: int = 0

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
        return float(self.bias_currents[self.bias_current_index])

    @property
    def power_dbm(self) -> float:
        return float(self.power_dbm_values[self.power_index]) if self.synthesizer.output else np.nan

    @property
    def frequency(self) -> float:
        return float(self.frequency_values[self.frequency_index]) if self.synthesizer.output else np.nan

    @property
    @abc.abstractmethod
    def stat_file(self) -> Path: ...

    @property
    @abc.abstractmethod
    def _line_index(self) -> int: ...

    @property
    @abc.abstractmethod
    def _line_name(self) -> str: ...

    @property
    def plot_line(self) -> pg.PlotDataItem:
        i: int = self._line_index
        if i not in self.plot_lines:
            color: QColor = pg.intColor(i)
            self.plot_lines[i] = self.figure.plot(np.empty(0), symbol='o', name=self._line_name,
                                                  pen=color, symbolPen=color, symbolBrush=color)
        return self.plot_lines[i]

    def start_measurement(self) -> None:
        if self.measurement is not None and self.measurement.is_alive():
            self.measurement.terminate()
            self.measurement.join()
        self.measurement = DetectMeasurement(results_queue=self.results_queue, state_queue=self.state_queue,
                                             good_to_go=self.good_to_measure,
                                             resistance=self.r,
                                             resistance_in_series=self.r_series,
                                             current_divider=self.divider,
                                             current_setting_function=self.setting_function,
                                             initial_biases=self.initial_biases,
                                             cycles_count=self.cycles_count,
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
        print(f'saving to {self.stat_file}')
        self.timer.start(50)

    @abc.abstractmethod
    def _next_indices(self) -> bool: ...

    def on_button_start_clicked(self) -> None:
        super(DetectBase, self).on_button_start_clicked()
        self.plot_line.clear()

        self.synthesizer.pulse_modulation.source = 'ext'
        self.synthesizer.pulse_modulation.state = True
        self.synthesizer.output = True

        while self.check_exists and self.stat_file.exists():
            warning(f'{self.stat_file} already exists')
            if not self._next_indices():
                error('nothing left to measure')
                self.synthesizer.pulse_modulation.state = False
                self.synthesizer.output = False
                return

        self.triton.issue_temperature(6, self.temperature)
        self.label_temperature.setValue(self.temperature * 1000)
        self.synthesizer.frequency = self.frequency * 1e9
        self.label_frequency.setValue(self.frequency)
        self.label_bias.setValue(self.bias_current)
        self.synthesizer.power.level = self.power_dbm
        self.label_power.setValue(self.power_dbm)
        self.start_measurement()

    def on_button_stop_clicked(self) -> None:
        if self.measurement is not None:
            self.measurement.terminate()
            self.measurement.join()
        self.timer.stop()
        self.synthesizer.pulse_modulation.state = False
        self.synthesizer.output = False
        super(DetectBase, self).on_button_stop_clicked()

    @abc.abstractmethod
    def on_timeout(self) -> None: ...


if __name__ == '__main__':
    app: QApplication = QApplication(sys.argv)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    window: DetectBase = DetectBase()
    window.show()
    app.exec()
    zero_sources()
