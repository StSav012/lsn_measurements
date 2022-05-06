# -*- coding: utf-8 -*-
import time
from datetime import datetime, timedelta
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Final, List, Literal, Sequence, Tuple, cast

import numpy as np
from nidaqmx.constants import *
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.task import Task
from numpy.typing import NDArray

from backend.hardware import *
from backend.utils import FileWriter, Count, measure_offsets, error, sine_segments, linear_segments

__all__ = ['LifetimeMeasurement']


class LifetimeMeasurement(Process):
    def __init__(self,
                 results_queue: 'Queue[Tuple[float, float, float]]',
                 state_queue: 'Queue[Tuple[int, timedelta]]',
                 good_to_go: SharedMemory,
                 *,
                 voltage_gain: float,
                 current_divider: float,
                 resistance: float,

                 bias_current: float,
                 initial_biases: Sequence[float],
                 current_setting_function: str,
                 setting_time: float,

                 trigger_voltage: float,
                 cycles_count: int,
                 ignore_never_switched: bool,

                 stat_file: Path,
                 data_file: Path,

                 power_dbm: float = np.nan,
                 frequency: float = np.nan,
                 resistance_in_series: float = 0.0,
                 max_waiting_time: timedelta = timedelta.max,
                 delay_between_cycles: float = 0.0,
                 max_reasonable_bias_error: float = np.inf,
                 temperature: float = np.nan) -> None:
        super(LifetimeMeasurement, self).__init__()

        self.results_queue: Queue[Tuple[float, float, float]] = results_queue
        self.state_queue: Queue[Tuple[int, timedelta]] = state_queue
        self.good_to_go: SharedMemory = SharedMemory(name=good_to_go.name)

        self.gain: Final[float] = voltage_gain
        self.divider: Final[float] = current_divider
        self.r: Final[float] = resistance
        self.r_series: Final[float] = resistance_in_series

        self.bias_current: Final[float] = bias_current
        self.initial_biases: List[float] = list(initial_biases)
        self.reset_function: Final[str] = current_setting_function
        self.setting_time: Final[float] = setting_time

        self.power_dbm: Final[float] = power_dbm
        self.frequency: Final[float] = frequency

        self.trigger_voltage: float = trigger_voltage
        self.cycles_count: Final[int] = cycles_count
        self.max_waiting_time: Final[timedelta] = max_waiting_time
        self.delay_between_cycles: Final[float] = delay_between_cycles
        self.max_reasonable_bias_error: Final[float] = max_reasonable_bias_error

        self.ignore_never_switched: Final[bool] = ignore_never_switched

        self.temperature: Final[float] = temperature

        self.stat_file: Path = stat_file
        self.data_file: Path = data_file

        self.c: Count = Count()

    def run(self) -> None:
        self.trigger_voltage -= offsets[adc_voltage.name]
        measure_offsets(0.04, do_zero_sources=True)
        self.trigger_voltage += offsets[adc_voltage.name]

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

            bias_current_steps_count: int = round(self.setting_time * task_dac.timing.samp_clk_max_rate)

            trigger_trigger: float = 0.45 * sync_channel.ao_max
            trigger_on_sequence: np.ndarray = np.zeros(bias_current_steps_count)
            trigger_on_sequence[-1] = 2.0 * trigger_trigger
            trigger_off_sequence: np.ndarray = np.zeros(bias_current_steps_count)

            task_adc.timing.cfg_samp_clk_timing(rate=task_adc.timing.samp_clk_max_rate,
                                                sample_mode=AcquisitionType.CONTINUOUS,
                                                samps_per_chan=10000,
                                                )
            task_dac.timing.cfg_samp_clk_timing(rate=task_dac.timing.samp_clk_max_rate,
                                                sample_mode=AcquisitionType.FINITE,
                                                samps_per_chan=bias_current_steps_count,
                                                )

            adc_stream: AnalogMultiChannelReader = AnalogMultiChannelReader(task_adc.in_stream)

            def reading_task_callback(_task_idx: int, _event_type: int, num_samples: int, _callback_data: Any) \
                    -> Literal[0]:
                data: np.ndarray = np.empty((3, num_samples))
                adc_stream.read_many_sample(data, num_samples)
                waiting: np.ndarray = (data[2] > trigger_trigger)
                data[1] -= self.r_series / self.r * data[0] * self.gain
                not_switched: np.ndarray = (data[1, waiting] < self.trigger_voltage)
                # fw.write_data(Path(r'd:\ttt\v(t)'), 'at', data)
                if np.any(waiting):
                    # fw.write_data(r'd:\ttt\voltage at 'f'{task_adc.timing.samp_clk_rate} Ss',
                    #               'at', data[1, waiting])
                    if self.c.loadable and not self.c.loaded:
                        this_time_not_switched: int = np.count_nonzero(not_switched)
                        self.c.inc(this_time_not_switched)
                        if not_switched.size > this_time_not_switched:
                            trig_arg: int = np.argwhere(data[1] > self.trigger_voltage).ravel()[0]
                            self.c.payload = (data[0, trig_arg],
                                              data[1, trig_arg],
                                              int(self.c) / task_adc.timing.samp_clk_rate)
                            self.c.loaded = True
                            self.c.reset()
                else:
                    self.c.loadable = np.any(data[1, ~waiting] < 0.5 * self.trigger_voltage)
                return 0

            # noinspection PyTypeChecker
            task_adc.register_every_n_samples_acquired_into_buffer_event(task_adc.timing.samp_quant_samp_per_chan,
                                                                         reading_task_callback)

            task_adc.start()

            task_dac.write(np.row_stack((
                np.full(bias_current_steps_count,
                        self.initial_biases[-1] * 1e-9 * self.divider * (DIVIDER_RESISTANCE + self.r)),
                trigger_off_sequence
            )), auto_start=True)
            task_dac.wait_until_done()
            task_dac.stop()

            measurement_start_time: datetime = datetime.now()

            bias_current_amplitude: float = np.abs(float(self.bias_current) - self.initial_biases[-1])
            actual_bias_current_steps_count: int = \
                round(bias_current_amplitude * 1e-9 * self.r * self.divider / (20 / 65536))
            actual_bias_current_step: float = bias_current_amplitude / (actual_bias_current_steps_count - 1)

            print(f'\nbias current is set to {self.bias_current} nA')
            print(f'number of current steps is {actual_bias_current_steps_count}')
            print(f'current step is {actual_bias_current_step:.4f} nA\n')
            if actual_bias_current_step > 4e-3 * bias_current_amplitude:
                error('current steps are too large')
                return

            # calculate the current sequence
            i_set: np.ndarray
            i_unset: np.ndarray
            if self.reset_function.casefold() == 'sine':
                i_set = sine_segments([self.initial_biases[-1], float(self.bias_current)],
                                      bias_current_steps_count) * self.r * self.divider
                i_unset = sine_segments([float(self.bias_current)] + self.initial_biases,
                                        bias_current_steps_count) * self.r * self.divider
            elif self.reset_function.casefold() == 'linear':
                i_set = linear_segments([self.initial_biases[-1], float(self.bias_current)],
                                        bias_current_steps_count) * self.r * self.divider
                i_unset = linear_segments([float(self.bias_current)] + self.initial_biases,
                                          bias_current_steps_count) * self.r * self.divider
            else:
                raise ValueError('Unsupported current setting function:', self.reset_function)

            i_set = np.row_stack((i_set * (1. + DIVIDER_RESISTANCE / self.r) * 1e-9, trigger_on_sequence))
            i_unset = np.row_stack((i_unset * (1. + DIVIDER_RESISTANCE / self.r) * 1e-9, trigger_off_sequence))

            task_dac.write(i_set, auto_start=True)
            task_dac.wait_until_done()
            task_dac.stop()
            task_dac.write(i_unset, auto_start=True)
            task_dac.wait_until_done()
            task_dac.stop()

            switching_time: np.ndarray = np.full(self.cycles_count, np.nan)
            set_bias_current: np.ndarray = np.full(self.cycles_count, np.nan)

            for cycle_index in range(self.cycles_count):
                while not self.good_to_go.buf[0]:
                    time.sleep(1)

                # set initial state
                task_dac.write(np.row_stack((
                    np.full(bias_current_steps_count,
                            (min(0.0, self.initial_biases[-1] * 1e-9 * self.r * self.divider)
                             * (1. + DIVIDER_RESISTANCE / self.r))),
                    np.zeros(bias_current_steps_count))),
                        auto_start=True)
                task_dac.wait_until_done()
                task_dac.stop()
                while not self.c.loadable:
                    time.sleep(0.01)
                self.c.loaded = False

                print(f'cycle {cycle_index + 1} out of {self.cycles_count}:', end=' ')

                # set bias
                task_dac.write(i_set, auto_start=True)
                task_dac.wait_until_done()
                task_dac.stop()

                t0: datetime = datetime.now()
                t1: datetime = datetime.now()

                while t1 - t0 <= self.max_waiting_time and not self.c.loaded:
                    time.sleep(0.01)
                    self.state_queue.put((cycle_index, t1 - t0))
                    t1 = datetime.now()

                if self.c.loaded:
                    self.c.loadable = False
                    i, v, t = self.c.payload
                    v = (v - offsets[adc_voltage.name]) / self.gain
                    i = (i - v - offsets[adc_current.name]) / self.r

                    set_bias_current[cycle_index] = i
                    switching_time[cycle_index] = t
                    print(f'switching at t = {t:.5f} s, {i * 1e9} nA, {v * 1e3} mV')
                    self.state_queue.put((cycle_index, timedelta(seconds=t)))
                    fw.write(self.data_file, 'at',
                             [self.frequency, self.bias_current, t, i * 1e9, v * 1e3,
                              (datetime.now() - measurement_start_time).total_seconds()])

                    self.c.loaded = False
                else:
                    print('no switching events detected')
                    if not self.ignore_never_switched:
                        i, v = np.nan, np.nan
                        switching_time[cycle_index] = self.max_waiting_time.total_seconds()
                        self.state_queue.put((cycle_index, self.max_waiting_time))
                        fw.write(self.data_file, 'at',
                                 [self.bias_current, self.max_waiting_time.total_seconds(), i * 1e9, v * 1e3,
                                  (datetime.now() - measurement_start_time).total_seconds()])

                task_dac.write(i_unset, auto_start=True)
                task_dac.wait_until_done()
                task_dac.stop()

                time.sleep(self.delay_between_cycles)

            if np.count_nonzero(~np.isnan(switching_time)) > 1:
                print(f'for bias current set to {self.bias_current} nA, '
                      f'mean switching time is {np.nanmean(switching_time)} s '
                      f'± {np.nanstd(switching_time)} s')
                # set_bias_current = (set_bias_current - offsets[adc_current.name]) / r
                # noinspection PyTypeChecker
                median_bias_current: float = np.nanmedian(set_bias_current)
                min_reasonable_bias_current: float = \
                    median_bias_current * (1. - .01 * self.max_reasonable_bias_error)
                max_reasonable_bias_current: float = \
                    median_bias_current * (1. + .01 * self.max_reasonable_bias_error)
                reasonable: np.ndarray = ((set_bias_current >= min_reasonable_bias_current)
                                          & (set_bias_current <= max_reasonable_bias_current))
                switching_time_reasonable: NDArray[np.float64] = switching_time[reasonable]
                non_zero: np.ndarray = (switching_time_reasonable > 0.0)
                switching_time_rnz: NDArray[np.float64] = switching_time_reasonable[non_zero]
                if not self.stat_file.exists():
                    self.stat_file.write_text('\t'.join((
                        'Frequency [GHz]',
                        'Set Bias Current [nA]',
                        'Mean Bias Current [nA]',
                        'Bias Current StD [nA]',

                        'τ₀ [s]',
                        'σ₀ [s]',

                        'τ [s]',
                        'σ [s]',

                        'τ₀/σ₀',
                        'τ/σ',
                    )) + '\n', encoding='utf-8')
                with self.stat_file.open('at', encoding='utf-8') as f_out:
                    f_out.write('\t'.join((
                        f'{self.frequency:.10f}'.rstrip('0').rstrip('.'),
                        f'{self.bias_current:.10f}'.rstrip('0').rstrip('.'),
                        f'{np.nanmean(set_bias_current[reasonable]) * 1e9:.10f}'
                        if np.any(reasonable) else 'nan',
                        f'{np.nanstd(set_bias_current[reasonable]) * 1e9:.10f}'
                        if np.any(reasonable) else 'nan',

                        f'{np.nanmean(switching_time_reasonable):.10f}'
                        if np.any(reasonable) else 'nan',
                        f'{np.nanstd(switching_time_reasonable):.10f}'
                        if np.any(reasonable) else 'nan',

                        f'{np.nanmean(switching_time_rnz):.10f}'
                        if np.any(reasonable) and np.any(non_zero) else 'nan',
                        f'{np.nanstd(switching_time_rnz):.10f}'
                        if np.any(reasonable) and np.any(non_zero) else 'nan',

                        f'{np.nanmean(switching_time_reasonable) / np.nanstd(switching_time_reasonable):.10f}'
                        if np.any(reasonable) else 'nan',
                        f'{np.nanmean(switching_time_rnz) / np.nanstd(switching_time_rnz):.10f}'
                        if np.any(reasonable) else 'nan',
                    )) + '\n')
                self.results_queue.put((cast(float, np.nanmean(set_bias_current[reasonable]) * 1e9),
                                        cast(float, np.nanmean(switching_time[reasonable])),
                                        cast(float, np.nanmean(switching_time[reasonable][non_zero]))))
            else:
                print(f'no switching event detected for bias current set to {self.bias_current} nA')