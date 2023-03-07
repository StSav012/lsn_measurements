# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from datetime import datetime, timedelta
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Final, List, Literal, Optional, Sequence, Tuple, cast

import numpy as np
from nidaqmx.constants import *
from nidaqmx.errors import DaqReadError
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.task import Task
from numpy.typing import NDArray

from backend.hardware import *
from backend.utils import FileWriter, PrintQueue, error, linear_segments, measure_offsets, sine_segments
from backend.utils.string_utils import format_float

fw: FileWriter = FileWriter()
fw.start()

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

        task_adc: Task
        task_dac: Task
        with Task() as task_adc, Task() as task_dac:
            task_adc.ai_channels.add_ai_voltage_chan(adc_current.name)
            task_adc.ai_channels.add_ai_voltage_chan(adc_voltage.name)
            task_adc.ai_channels.add_ai_voltage_chan(adc_sync.name)
            current_channel = task_dac.ao_channels.add_ao_voltage_chan(dac_current.name)
            sync_channel = task_dac.ao_channels.add_ao_voltage_chan(dac_sync.name)

            bias_current_amplitude: float = np.abs(self.max_bias_current - self.initial_biases[-1])
            dac_rate: float = task_dac.timing.samp_clk_max_rate
            bias_current_steps_count: int = round(bias_current_amplitude / self.current_speed * dac_rate)
            samples_per_dac_channel: int = 2 * bias_current_steps_count
            if samples_per_dac_channel > task_dac.output_onboard_buffer_size:
                dac_rate /= samples_per_dac_channel / task_dac.output_onboard_buffer_size
                bias_current_steps_count = round(bias_current_amplitude / self.current_speed * dac_rate)
                samples_per_dac_channel = 2 * bias_current_steps_count
            # If we get too many samples per channel again, we sacrifice the current steps
            while samples_per_dac_channel > task_dac.output_onboard_buffer_size:
                bias_current_steps_count -= 1
                samples_per_dac_channel = 2 * bias_current_steps_count

            actual_bias_current_steps_count: int = round(bias_current_amplitude * 1e-9
                                                         * self.r * self.divider
                                                         / min(((current_channel.ao_max - current_channel.ao_min)
                                                                / (2 ** current_channel.ao_resolution)),
                                                               bias_current_steps_count))
            actual_bias_current_step: float = bias_current_amplitude / (actual_bias_current_steps_count - 1)

            pq.write(f'\nnumber of current steps = {actual_bias_current_steps_count}')
            pq.write(f'current step = {actual_bias_current_step:.4f} nA\n')
            if actual_bias_current_step > 4e-3 * bias_current_amplitude:
                error('current steps are too large')
                return

            switching_current: NDArray[np.float64] = np.full(self.cycles_count, np.nan)
            switching_voltage: NDArray[np.float64] = np.full(self.cycles_count, np.nan)

            trigger_trigger: float = 0.45 * sync_channel.ao_max
            trigger_sequence: NDArray[np.float64] = np.concatenate((
                np.full(bias_current_steps_count, 2. * trigger_trigger),
                np.zeros(bias_current_steps_count)
            ))

            task_adc.timing.cfg_samp_clk_timing(rate=task_adc.timing.samp_clk_max_rate,
                                                sample_mode=AcquisitionType.CONTINUOUS,
                                                samps_per_chan=1000,
                                                )
            task_dac.timing.cfg_samp_clk_timing(rate=dac_rate,
                                                sample_mode=AcquisitionType.FINITE,
                                                samps_per_chan=samples_per_dac_channel,
                                                )

            adc_stream: AnalogMultiChannelReader = AnalogMultiChannelReader(task_adc.in_stream)

            def reading_task_callback(_task_idx: int, _event_type: int, num_samples: int, _callback_data: Any) \
                    -> Literal[0]:
                data: NDArray[np.float64] = np.empty((3, num_samples), dtype=np.float64)
                try:
                    adc_stream.read_many_sample(data, num_samples)
                except DaqReadError as ex:
                    self.pulse_ended = True
                    self.pulse_started = False
                    pq.write(repr(ex), end='')
                increasing_current: NDArray[np.bool] = (data[2] > trigger_trigger)
                if np.any(increasing_current):
                    data = data[0:2, increasing_current]
                    self.pulse_started = True
                    self.pulse_ended = not increasing_current[-1]
                    if np.isnan(switching_current[-1]):
                        v: NDArray[np.float64] = data[1]
                        v = (v - offsets[adc_voltage.name]) / self.gain
                        v -= self.r_series / self.r * data[0]
                        _switching_events_count: int
                        if (v[0] < self.trigger_voltage) and (self.trigger_voltage < v[-1]):
                            if not self.ignore_switch:
                                i: NDArray[np.float64] = (data[0] - v - offsets[adc_current.name]) / self.r
                                _index: int = cast(int, np.searchsorted(v, self.trigger_voltage))

                                _switching_events_count = np.count_nonzero(~np.isnan(switching_current))
                                switching_current[_switching_events_count] = i[_index]
                                switching_voltage[_switching_events_count] = v[_index]
                                fw.write(self.data_file, 'at', (i[_index], v[_index], datetime.now().timestamp()))
                                self.switching_data_queue.put((i[_index], v[_index]))
                                pq.write(f'switching current is {i[_index] * 1e9:.2f} nA '
                                         f'(voltage is {v[_index] * 1e6:.3f} uV)',
                                         end='')
                            self.switch_registered = True
                        elif not self.switch_registered and v[0] > self.trigger_voltage:
                            if not self.ignore_switch:
                                _v: np.float64 = v[0]
                                _i: np.float64 = (data[0, 0] - _v - offsets[adc_current.name]) / self.r

                                _switching_events_count = np.count_nonzero(~np.isnan(switching_current))
                                switching_current[_switching_events_count] = _i
                                switching_voltage[_switching_events_count] = _v
                                fw.write(self.data_file, 'at', (_i, _v, datetime.now().timestamp()))
                                self.switching_data_queue.put((_i, _v))
                                pq.write(f'switching current is {_i * 1e9:.2f} nA '
                                         f'(voltage is {_v * 1e6:.3f} uV)',
                                         end='')
                            self.switch_registered = True
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
            i_set: NDArray[np.float64] = np.row_stack((np.concatenate((
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
                reasonable: NDArray[np.float64] = ((switching_current >= min_reasonable_switching_current)
                                                   & (switching_current <= max_reasonable_switching_current))
                reasonable_switching_current: NDArray[np.float64] = switching_current[reasonable]
                mean_switching_current: np.float64 | NDArray[np.float64]
                switching_current_std: np.float64 | NDArray[np.float64]
                if reasonable_switching_current.size:
                    mean_switching_current = 1e9 * np.nanmean(reasonable_switching_current)
                    switching_current_std = 1e9 * np.nanstd(reasonable_switching_current)
                else:
                    mean_switching_current = np.nan
                    switching_current_std = np.nan
                if not self.stat_file.exists():
                    self.stat_file.write_text('\t'.join((
                        'Temperature [mK]',
                        'Frequency [GHz]',
                        'Power [dBm]',
                        'Mean Switch Current [nA]',
                        'Switch Current StD [nA]',
                        'Measurement Duration [s]',

                        'Actual Temperature [mK]',
                    )) + '\n', encoding='utf-8')
                with self.stat_file.open('at', encoding='utf-8') as f_out:
                    f_out.write('\t'.join((
                        format_float(self.temperature * 1000),
                        format_float(self.frequency),
                        format_float(self.power_dbm),
                        f'{mean_switching_current:.6f}',
                        f'{switching_current_std:.6f}',
                        f'{(datetime.now() - measurement_start_time).total_seconds():.3f}',

                        bytes(self.good_to_go.buf[1:128]).strip(b'\0').decode(),
                    )) + '\n')
                self.results_queue.put((cast(float, mean_switching_current),
                                        cast(float, switching_current_std)))
