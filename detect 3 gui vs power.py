# -*- coding: utf-8 -*-
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


class GUI(DetectGUI):
    def setup_ui_appearance(self) -> None:
        super(GUI, self).setup_ui_appearance()

        self.figure.getAxis('bottom').setLabel(text='Power', units='dBm')


class App(GUI):
    def __init__(self, flags=Qt.WindowFlags()) -> None:
        super(App, self).__init__(flags=flags)

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
    def stat_file(self) -> Path:
        return self.saving_location / (' '.join((
            'detect',
            self.config.get('output', 'prefix', fallback=''),
            f'{self.temperature * 1e3:.6f}'.rstrip('0').rstrip('.') + 'mK',
            f'{self.bias_current:.6f}'.rstrip('0').rstrip('.') + 'nA',
            f'CC{self.cycles_count}',
            f'{self.frequency:.6f}'.rstrip('0').rstrip('.') + 'GHz'
            if self.synthesizer.output else '',
            f'P{self.pulse_duration:.6f}'.rstrip('0').rstrip('.') + 's',
            f'WaP{self.waiting_after_pulse:.6f}'.rstrip('0').rstrip('.') + 's',
            f'ST{self.setting_time:.6f}'.rstrip('0').rstrip('.') + 's',
            self.config.get('output', 'suffix', fallback='')
        )).replace('  ', ' ').strip(' ') + '.txt')

    @property
    def _line_index(self) -> int:
        return (self.bias_current_index
                + (self.frequency_index * len(self.bias_currents)
                   + self.temperature_index) * len(self.frequency_values))

    @property
    def _line_name(self) -> str:
        return ', '.join((
            f'{self.bias_current:.6f}'.rstrip('0').rstrip('.') + 'nA',
            f'{self.temperature * 1e3:.6f}'.rstrip('0').rstrip('.') + 'mK',
            f'{self.frequency:.6f}'.rstrip('0').rstrip('.') + 'GHz'
            if not np.isnan(self.frequency) else '',
        )).replace('  ', ' ').replace('  ', ' ').strip(', ')

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

    def on_button_start_clicked(self) -> None:
        super(App, self).on_button_start_clicked()
        self.plot_line.clear()

        self.synthesizer.pulse_modulation.source = 'ext'
        self.synthesizer.pulse_modulation.state = True
        self.synthesizer.output = True

        while self.check_exists and self.stat_file.exists():
            warning(f'{self.stat_file} already exists')
            self.bias_current_index += 1
            if self.bias_current_index >= len(self.bias_currents):
                self.bias_current_index = 0
                self.frequency_index += 1
                if self.frequency_index >= len(self.frequency_values):
                    self.frequency_index = 0
                    self.temperature_index += 1
                    if self.temperature_index >= len(self.temperature_values):
                        self.temperature_index = 0
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
        super(App, self).on_button_stop_clicked()

    def on_timeout(self) -> None:
        cycle_index: int
        estimated_cycles_count: int
        switches_count: int
        while not self.state_queue.empty():
            cycle_index, estimated_cycles_count, switches_count = self.state_queue.get(block=True)
            self.label_loop_number.setValue(cycle_index + 1)
            self.label_loop_count.setValue(estimated_cycles_count)
            self.label_probability.setValue(switches_count / (cycle_index + 1) * 100)

        prob: float = np.inf
        err: float
        while not self.results_queue.empty():
            old_x_data: np.ndarray = np.empty(0) if self.plot_line.xData is None else self.plot_line.xData
            old_y_data: np.ndarray = np.empty(0) if self.plot_line.yData is None else self.plot_line.yData
            prob, err = self.results_queue.get(block=True)
            x_data: np.ndarray = np.concatenate((old_x_data, [self.power_dbm]))
            y_data: np.ndarray = np.concatenate((old_y_data, [prob]))
            self.plot_line.setData(x_data, y_data)

        actual_temperature: float
        temperature_unit: str
        actual_temperature, temperature_unit = self.triton.query_temperature(6)
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
            td: timedelta = datetime.now() - self.bad_temperature_time
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

        self.good_to_measure.buf[0] &= not self.button_pause.isChecked()

        if not self.measurement.is_alive():
            self.timer.stop()
            if self.stop_key_power.isChecked():
                self.on_button_stop_clicked()
                return
            self.power_index += 1
            if prob < self.minimal_probability_to_measure or self.power_index == len(self.power_dbm_values):
                self.power_index = 0
                if self.stop_key_bias.isChecked():
                    self.on_button_stop_clicked()
                    return
                if self.check_exists:
                    while self.bias_current_index < len(self.bias_currents) and self.stat_file.exists():
                        self.bias_current_index += 1
                else:
                    self.bias_current_index += 1
                if self.bias_current_index >= len(self.bias_currents):
                    self.bias_current_index = 0
                    if self.stop_key_frequency.isChecked():
                        self.on_button_stop_clicked()
                        return
                    if self.check_exists:
                        while self.frequency_index < len(self.frequency_values) and self.stat_file.exists():
                            self.frequency_index += 1
                    else:
                        self.frequency_index += 1
                    if self.frequency_index >= len(self.frequency_values):
                        self.frequency_index = 0
                        if self.stop_key_temperature.isChecked():
                            self.on_button_stop_clicked()
                            return
                        if self.check_exists:
                            while self.temperature_index < len(self.temperature_values) and self.stat_file.exists():
                                self.temperature_index += 1
                        else:
                            self.temperature_index += 1
                        if self.temperature_index >= len(self.temperature_values):
                            self.temperature_index = 0
                            self.on_button_stop_clicked()
                            return
                        self.triton.issue_temperature(6, self.temperature)
                        self.label_temperature.setValue(self.temperature * 1000)
                    self.synthesizer.frequency = self.frequency * 1e9
                    self.label_frequency.setValue(self.frequency)
                self.label_bias.setValue(self.bias_current)
            self.synthesizer.power.level = self.power_dbm
            self.label_power.setValue(self.power_dbm)

            self.start_measurement()
        else:
            self.timer.setInterval(50)


if __name__ == '__main__':
    app: QApplication = QApplication(sys.argv)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    window: App = App()
    window.show()
    app.exec()
    zero_sources()
