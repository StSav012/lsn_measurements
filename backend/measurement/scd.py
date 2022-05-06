# -*- coding: utf-8 -*-
import time
from datetime import datetime, timedelta
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Final, List, Literal, Optional, Sequence, Tuple, cast

import numpy as np
from nidaqmx.constants import *
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.task import Task
from numpy.typing import NDArray

from backend.hardware import *
from backend.utils import PrintQueue, FileWriter, error, linear_segments, sine_segments, measure_offsets

pq: PrintQueue = PrintQueue()
pq.start()

__all__ = ['SCDMeasurement']


class SCDMeasurement(Process):
    def __init__(self,
                 results_queue: 'Queue[Tuple[float, float]]',
                 state_queue: 'Queue[Tuple[int, timedelta]]',
                 switching_data_queue: 'Queue[Tuple[np.float64, np.float64]]',
                 good_to_go: SharedMemory,
                 *,
                 voltage_gain: float,
                 current_divider: float,
                 resistance: float,

                 max_bias_current: float,
                 initial_biases: Sequence[float],
                 current_reset_function: str,
                 current_speed: float,

                 trigger_voltage: float,
                 cycles_count: int,

                 stat_file: Path,
                 data_file: Path,

                 power_dbm: float = np.nan,
                 frequency: float = np.nan,
                 resistance_in_series: float = 0.0,
                 max_measurement_time: timedelta = timedelta.max,
                 delay_between_cycles: float = 0.0,
                 max_reasonable_bias_error: float = np.inf,
                 temperature: float = np.nan) -> None:
        super(SCDMeasurement, self).__init__()

        self.results_queue: Queue[Tuple[float, float]] = results_queue
        self.state_queue: Queue[Tuple[int, Optional[timedelta]]] = state_queue
        self.switching_data_queue: Queue[Tuple[np.float64, np.float64]] = switching_data_queue
        self.good_to_go: SharedMemory = SharedMemory(name=good_to_go.name)

        self.gain: Final[float] = voltage_gain
        self.divider: Final[float] = current_divider
        self.r: Final[float] = resistance
        self.r_series: Final[float] = resistance_in_series

        self.max_bias_current: Final[float] = max_bias_current
        self.initial_biases: List[float] = list(initial_biases)
        self.reset_function: Final[str] = current_reset_function
        self.current_speed: Final[float] = current_speed

        self.power_dbm: Final[float] = power_dbm
        self.frequency: Final[float] = frequency

        self.trigger_voltage: float = trigger_voltage
        self.cycles_count: Final[int] = cycles_count
        self.max_measurement_time: Final[timedelta] = max_measurement_time
        self.delay_between_cycles: Final[float] = delay_between_cycles
        self.max_reasonable_bias_error: Final[float] = max_reasonable_bias_error

        self.temperature: Final[float] = temperature

        self.stat_file: Path = stat_file
        self.data_file: Path = data_file

        self.pulse_started: bool = False
        self.pulse_ended: bool = False
        self.switch_registered: bool = False
        self.ignore_switch: bool = False

    def run(self) -> None:
        measure_offsets()

        fw: FileWriter = FileWriter()
        fw.start()

        task_adc: Task
        task_dac: Task
        with Task() as task_adc, Task() as task_dac:
            task_adc.ai_channels.add_ai_voltage_chan(adc_current.name)
            task_adc.ai_channels.add_ai_voltage_chan(adc_voltage.name)
            task_adc.ai_channels.add_ai_voltage_chan(adc_sync.name)
            task_dac.ao_channels.add_ao_voltage_chan(dac_current.name)
            sync_channel = task_dac.ao_channels.add_ao_voltage_chan(dac_sync.name)

            bias_current_amplitude: float = np.abs(self.max_bias_current - self.initial_biases[-1])
            bias_current_steps_count: int = round(
                bias_current_amplitude / self.current_speed * task_dac.timing.samp_clk_max_rate)
            actual_bias_current_steps_count: int = \
                round(bias_current_amplitude * 1e-9 * self.r * self.divider / (20 / 65536))
            actual_bias_current_step: float = bias_current_amplitude / (actual_bias_current_steps_count - 1)

            pq.write(f'\nnumber of current steps = {actual_bias_current_steps_count}')
            pq.write(f'current step = {actual_bias_current_step:.4f} nA\n')
            if actual_bias_current_step > 4e-3 * bias_current_amplitude:
                error('current steps are too large')
                return

            switching_current: np.ndarray = np.full(self.cycles_count, np.nan)
            switching_voltage: np.ndarray = np.full(self.cycles_count, np.nan)

            trigger_trigger: float = 0.45 * sync_channel.ao_max
            trigger_sequence: np.ndarray = np.concatenate((
                np.full(bias_current_steps_count, 2. * trigger_trigger),
                np.zeros(bias_current_steps_count)
            ))

            task_adc.timing.cfg_samp_clk_timing(rate=task_adc.timing.samp_clk_max_rate,
                                                sample_mode=AcquisitionType.CONTINUOUS,
                                                samps_per_chan=10000,
                                                )
            task_dac.timing.cfg_samp_clk_timing(rate=task_dac.timing.samp_clk_max_rate,
                                                sample_mode=AcquisitionType.FINITE,
                                                samps_per_chan=2 * bias_current_steps_count,
                                                )

            adc_stream: AnalogMultiChannelReader = AnalogMultiChannelReader(task_adc.in_stream)

            def reading_task_callback(_task_idx: int, _event_type: int, num_samples: int, _callback_data: Any) \
                    -> Literal[0]:
                data: NDArray[np.float64] = np.empty((3, num_samples), dtype=np.float64)
                adc_stream.read_many_sample(data, num_samples)
                increasing_current: NDArray[np.bool] = (data[2] > trigger_trigger)
                if np.any(increasing_current):
                    data = data[0:2, increasing_current]
                    self.pulse_started = True
                    self.pulse_ended = not increasing_current[-1]
                    if np.isnan(switching_current[-1]):
                        v: np.ndarray = data[1]
                        v = (v - offsets[adc_voltage.name]) / self.gain
                        v -= self.r_series / self.r * data[0]
                        # if np.any(v < self.trigger_voltage) and np.any(self.trigger_voltage < v):
                        if (v[0] < self.trigger_voltage) and (self.trigger_voltage < v[-1]):
                            i: np.ndarray = (data[0] - v - offsets[adc_current.name]) / self.r
                            # _index: int = cast(int, np.argmin(np.abs(v - self.trigger_voltage)))
                            _index: int = cast(int, np.searchsorted(v, self.trigger_voltage))
                            # if data[2, _index] < trigger_trigger:  # if the current is decreasing
                            #     return 0

                            if not self.ignore_switch:
                                _switching_events_count: int = np.count_nonzero(~np.isnan(switching_current))
                                switching_current[_switching_events_count] = i[_index]
                                switching_voltage[_switching_events_count] = v[_index]
                                fw.write(self.data_file, 'at', (i[_index], v[_index]))
                                self.switching_data_queue.put((i[_index], v[_index]))
                                pq.write(f'switching current is {i[_index] * 1e9:.2f} nA '
                                         f'(voltage is {v[_index] * 1e6:.3f} uV)',
                                         end='')
                            self.switch_registered = True
                        # elif v[0] > trigger_voltage > v[-1]:
                        #     pq.write('switching to the superconducting branch')
                else:
                    if self.pulse_started:
                        self.pulse_ended = True
                        self.pulse_started = False
                        if not self.switch_registered:
                            pq.write('no switching events', end='')
                return 0

            # noinspection PyTypeChecker
            task_adc.register_every_n_samples_acquired_into_buffer_event(task_adc.timing.samp_quant_samp_per_chan,
                                                                         reading_task_callback)

            # calculating the current sequence
            i_set: np.ndarray = np.row_stack((np.concatenate((
                np.linspace(self.initial_biases[-1], self.max_bias_current, bias_current_steps_count, endpoint=True),
                {
                    'linear': linear_segments([self.max_bias_current] + self.initial_biases, bias_current_steps_count),
                    'sine': sine_segments([self.max_bias_current] + self.initial_biases, bias_current_steps_count),
                }[self.reset_function.casefold()]
            )) * self.r * self.divider * (1. + DIVIDER_RESISTANCE / self.r) * 1e-9, trigger_sequence))

            task_adc.start()

            task_dac.write(np.row_stack((np.full(2 * bias_current_steps_count,
                                                 self.initial_biases[-1] * 1e-9 * self.r * self.divider),
                                         trigger_sequence)),
                           auto_start=True)
            task_dac.wait_until_done(WAIT_INFINITELY)
            task_dac.stop()

            switching_current = np.full(self.cycles_count, np.nan)
            switching_voltage = np.full(self.cycles_count, np.nan)

            self.ignore_switch = True
            for iv_curve_number in range(min(2, self.cycles_count)):
                pq.write(f'iv curve measurement {iv_curve_number + 1} out of {min(2, self.cycles_count)}')
                t0: datetime = datetime.now()
                task_dac.write(i_set, auto_start=True)
                task_dac.wait_until_done(WAIT_INFINITELY)
                task_dac.stop()
                t1: datetime = datetime.now()
                pq.write(f'\nmeasurement took {(t1 - t0).total_seconds()} seconds '
                         f'({(t1 - t0).total_seconds() / bias_current_steps_count} seconds per point)')

            while not self.pulse_ended:
                time.sleep(0.01)
            self.pulse_ended = False
            switching_current = np.full(self.cycles_count, np.nan)
            switching_voltage = np.full(self.cycles_count, np.nan)
            self.ignore_switch = False

            measurement_start_time: datetime = datetime.now()

            cycle_index: int = 1
            this_time_cycles_count: int = self.cycles_count
            while cycle_index <= this_time_cycles_count:
                while not self.good_to_go.buf[0]:
                    time.sleep(1)

                pq.write(f'cycle {cycle_index} out of {this_time_cycles_count}:', end=' ')

                switching_events_count: int = np.count_nonzero(~np.isnan(switching_current))

                if switching_events_count == self.cycles_count:
                    break

                self.pulse_started = False
                self.pulse_ended = False
                self.switch_registered = False
                task_dac.write(i_set, auto_start=True)
                task_dac.wait_until_done(WAIT_INFINITELY)
                task_dac.stop()
                while not self.pulse_ended:
                    time.sleep(0.01)
                self.pulse_ended = False

                this_time_cycles_count = cycle_index + np.count_nonzero(np.isnan(switching_current))

                time.sleep(self.delay_between_cycles)
                now: datetime = datetime.now()

                remaining_time: timedelta = \
                    (now - measurement_start_time) / cycle_index * (this_time_cycles_count - cycle_index)
                remaining_time = min(remaining_time,
                                     self.max_measurement_time - (now - measurement_start_time))
                self.state_queue.put((cycle_index, remaining_time))
                if remaining_time.total_seconds() >= 0:
                    pq.write(f'; time left: {remaining_time}')
                else:
                    pq.write()

                if now - measurement_start_time > self.max_measurement_time:
                    break

                cycle_index += 1

            if np.count_nonzero(~np.isnan(switching_current)) > 1:
                # noinspection PyTypeChecker
                median_switching_current: float = np.nanmedian(switching_current)
                min_reasonable_switching_current: float = \
                    median_switching_current * (1. - .01 * self.max_reasonable_bias_error)
                max_reasonable_switching_current: float = \
                    median_switching_current * (1. + .01 * self.max_reasonable_bias_error)
                reasonable: np.ndarray = ((switching_current >= min_reasonable_switching_current)
                                          & (switching_current <= max_reasonable_switching_current))
                with self.stat_file.open('at') as f_out:
                    f_out.write(f'{self.temperature}\t'
                                f'{self.frequency}\t'
                                f'{self.power_dbm}\t'
                                f'{1e9 * np.nanmean(switching_current[reasonable]):.6f}\t'
                                f'{1e9 * np.nanstd(switching_current[reasonable]):.6f}\t'
                                f'{(datetime.now() - measurement_start_time).total_seconds():.3f}\n')
                self.results_queue.put((cast(float, 1e9 * np.nanmean(switching_current[reasonable])),
                                        cast(float, 1e9 * np.nanstd(switching_current[reasonable]))))