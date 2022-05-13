# -*- coding: utf-8 -*-
import time
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Final, List, Literal, Sequence, Tuple

import numpy as np
from nidaqmx.constants import *
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.task import Task
from numpy.typing import NDArray

from backend.hardware import *
from backend.utils import Count, FileWriter, error, linear_segments, measure_offsets, sine_segments

__all__ = ['DetectMeasurement']

fw: FileWriter = FileWriter()
fw.start()


class DetectMeasurement(Process):
    def __init__(self,
                 results_queue: 'Queue[Tuple[float, float]]',
                 state_queue: 'Queue[Tuple[int, int, int]]',
                 good_to_go: SharedMemory,
                 *,
                 voltage_gain: float,
                 current_divider: float,
                 resistance: float,

                 bias_current: float,
                 initial_biases: Sequence[float],
                 current_setting_function: str,
                 setting_time: float,

                 pulse_duration: float,

                 trigger_voltage: float,
                 cycles_count: int,
                 max_switching_events_count: int,

                 stat_file: Path,
                 data_file: Path,

                 power_dbm: float = np.nan,
                 frequency: float = np.nan,
                 resistance_in_series: float = 0.0,
                 waiting_after_pulse: float = 0.0,
                 temperature: float = np.nan) -> None:
        super(DetectMeasurement, self).__init__()

        self.results_queue: Queue[Tuple[float, float]] = results_queue
        self.state_queue: Queue[Tuple[int, int, int]] = state_queue
        self.good_to_go: SharedMemory = SharedMemory(name=good_to_go.name)

        self.voltage_gain: Final[float] = voltage_gain
        self.divider: Final[float] = current_divider
        self.r: Final[float] = resistance
        self.r_series: Final[float] = resistance_in_series

        self.bias_current: Final[float] = bias_current
        self.initial_biases: List[float] = list(initial_biases)
        self.setting_function: Final[str] = current_setting_function
        self.setting_time: Final[float] = setting_time

        self.pulse_duration: Final[float] = pulse_duration
        self.waiting_after_pulse: Final[float] = waiting_after_pulse
        self.power_dbm: Final[float] = power_dbm
        self.frequency: Final[float] = frequency

        self.trigger_voltage: float = trigger_voltage
        self.cycles_count: Final[int] = cycles_count
        self.max_switching_events_count: Final[int] = max_switching_events_count

        self.temperature: Final[float] = temperature

        self.stat_file: Path = stat_file
        self.data_file: Path = data_file

        self.c: Count = Count()
        self.pulse_started: bool = False
        self.pulse_ended: bool = False

    def run(self) -> None:
        self.trigger_voltage -= offsets[adc_voltage.name]
        measure_offsets()
        self.trigger_voltage += offsets[adc_voltage.name]

        task_adc: Task
        task_dac: Task
        with Task() as task_adc, Task() as task_dac:
            task_adc.ai_channels.add_ai_voltage_chan(adc_current.name)
            task_adc.ai_channels.add_ai_voltage_chan(adc_voltage.name)
            task_adc.ai_channels.add_ai_voltage_chan(adc_sync.name)
            task_dac.ao_channels.add_ao_voltage_chan(dac_current.name)
            task_dac.ao_channels.add_ao_voltage_chan(dac_attenuation.name)
            sync_channel = task_dac.ao_channels.add_ao_voltage_chan(dac_sync.name)

            dac_rate: float = 0.2 * task_dac.timing.samp_clk_max_rate

            bias_current_steps_count: int = round(self.setting_time * dac_rate)
            pulse_duration_points_count: int = round(self.pulse_duration * dac_rate)
            waiting_after_pulse_points_count: int = round(self.waiting_after_pulse * dac_rate)

            trigger_trigger: float = 0.45 * sync_channel.ao_max
            trigger_sequence: np.ndarray = np.concatenate((
                np.zeros(bias_current_steps_count),
                np.full(pulse_duration_points_count + waiting_after_pulse_points_count, 2. * trigger_trigger),
                np.zeros(bias_current_steps_count)
            ))

            task_adc.timing.cfg_samp_clk_timing(rate=task_adc.timing.samp_clk_max_rate,
                                                sample_mode=AcquisitionType.CONTINUOUS,
                                                samps_per_chan=10000,
                                                )
            task_dac.timing.cfg_samp_clk_timing(rate=dac_rate,
                                                sample_mode=AcquisitionType.FINITE,
                                                samps_per_chan=(2 * bias_current_steps_count
                                                                + pulse_duration_points_count
                                                                + waiting_after_pulse_points_count),
                                                )

            adc_stream: AnalogMultiChannelReader = AnalogMultiChannelReader(task_adc.in_stream)

            def reading_task_callback(_task_idx: int, _event_type: int, num_samples: int, _callback_data: Any) \
                    -> Literal[0]:
                data: NDArray[np.float64] = np.empty((3, num_samples), dtype=np.float64)
                adc_stream.read_many_sample(data, num_samples)
                waiting: NDArray[np.bool] = (data[2] > trigger_trigger)
                data[1] -= self.r_series / self.r * data[0] * self.voltage_gain
                not_switched: NDArray[np.bool] = (data[1] < self.trigger_voltage)
                if np.any(waiting):
                    self.pulse_started = True
                    self.pulse_ended = not waiting[-1]
                    self.c.inc(np.count_nonzero(waiting & not_switched))
                    if self.c.loadable and not self.c.loaded and np.any(data[1] > self.trigger_voltage):
                        trig_arg: int = np.argwhere(data[1] > self.trigger_voltage).ravel()[0]
                        self.c.payload = (data[0, trig_arg],
                                          data[1, trig_arg],
                                          int(self.c) / task_adc.timing.samp_clk_rate)
                        self.c.loaded = True
                        self.c.reset()
                else:
                    if self.pulse_started:
                        self.pulse_ended = True
                        self.pulse_started = False
                    self.c.reset()
                self.c.loadable = np.any(data[1] < 0.5 * self.trigger_voltage) and not np.any(waiting)
                return 0

            # noinspection PyTypeChecker
            task_adc.register_every_n_samples_acquired_into_buffer_event(task_adc.timing.samp_quant_samp_per_chan,
                                                                         reading_task_callback)

            task_adc.start()

            bias_current_amplitude: float = np.abs(float(self.bias_current) - self.initial_biases[-1])
            actual_bias_current_steps_count: int = round(bias_current_amplitude * 1e-9
                                                         * self.r * self.divider / (20 / 65536))
            actual_bias_current_step: float = bias_current_amplitude / (actual_bias_current_steps_count - 1)

            print(f'\nbias current is set to {self.bias_current} nA')
            print(f'number of current steps is {actual_bias_current_steps_count}')
            print(f'current step is {actual_bias_current_step:.4f} nA\n')
            if actual_bias_current_step > 4e-3 * bias_current_amplitude:
                error('current steps are too large')
                return

            # calculate the current sequence
            i_set: np.ndarray
            if self.setting_function.casefold() == 'sine':
                i_set = np.concatenate((
                    sine_segments([self.initial_biases[-1], self.bias_current], bias_current_steps_count),
                    np.full(pulse_duration_points_count + waiting_after_pulse_points_count, self.bias_current),
                    sine_segments([self.bias_current] + self.initial_biases, bias_current_steps_count)
                )) * 1e-9 * self.r * self.divider
            elif self.setting_function.casefold() == 'linear':
                i_set = np.concatenate((
                    linear_segments([self.initial_biases[-1], self.bias_current],
                                    bias_current_steps_count) * self.r * self.divider,
                    np.full(pulse_duration_points_count + waiting_after_pulse_points_count, self.bias_current),
                    linear_segments([self.bias_current] + self.initial_biases,
                                    bias_current_steps_count) * self.r * self.divider
                )) * 1e-9 * self.r * self.divider
            else:
                raise ValueError('Unsupported current setting function:', self.setting_function)

            am_voltage_sequence: np.ndarray = np.concatenate((
                np.full(bias_current_steps_count, 0.0),
                np.full(pulse_duration_points_count, 1.5),
                np.full(waiting_after_pulse_points_count, 0.0),
                np.full(bias_current_steps_count, 0.0)
            ))
            period_sequence: np.ndarray = np.row_stack((i_set * (1. + DIVIDER_RESISTANCE / self.r),
                                                        am_voltage_sequence,
                                                        trigger_sequence))

            # set initial state
            task_dac.write(np.row_stack((
                np.full(task_dac.timing.samp_quant_samp_per_chan,
                        (min(0.0, self.initial_biases[-1] * 1e-9 * self.r * self.divider)
                         * (1. + DIVIDER_RESISTANCE / self.r))),
                np.zeros(task_dac.timing.samp_quant_samp_per_chan),
                np.zeros(task_dac.timing.samp_quant_samp_per_chan),
            )), auto_start=True)
            task_dac.wait_until_done()
            task_dac.stop()

            switches_count: int = 0
            actual_cycles_count: int = 0

            for cycle_index in range(self.cycles_count):
                while not self.good_to_go.buf[0]:
                    time.sleep(1)

                estimated_cycles_count: int
                if switches_count == 0:
                    estimated_cycles_count = self.cycles_count
                else:
                    estimated_cycles_count = \
                        min(self.cycles_count,
                            round(self.max_switching_events_count * actual_cycles_count / switches_count))
                print(f'{self.temperature * 1000:.1f} mK | {self.bias_current:.2f} nA | '
                      f'{self.frequency:.4f} GHz | {self.power_dbm:.2f} dBm | '
                      f'cycle {cycle_index + 1} out of {estimated_cycles_count}:', end=' ')

                while not self.c.loadable:
                    time.sleep(0.01)
                self.c.loaded = False
                self.pulse_started = False
                self.pulse_ended = False

                # set bias
                task_dac.write(period_sequence, auto_start=True)
                task_dac.wait_until_done()
                task_dac.stop()

                while not self.c.loadable or not self.pulse_ended:
                    time.sleep(0.01)
                self.pulse_started = False
                self.pulse_ended = False
                if self.c.loaded:
                    i, v, t = self.c.payload
                    self.c.loaded = False

                    v = (v - offsets[adc_voltage.name]) / self.voltage_gain
                    i = (i - v - offsets[adc_current.name]) / self.r

                    switches_count += 1
                    print('switching at'f' t = {t:.5f} s, {i * 1e9:.4f} nA, {v * 1e3:.4f} mV')
                    fw.write(self.data_file, 'at', (i * 1e9, v * 1e3, t))
                else:
                    print('no switching events detected')
                    self.c.reset()

                self.state_queue.put((cycle_index, estimated_cycles_count, switches_count))

                actual_cycles_count = cycle_index + 1
                if switches_count >= self.max_switching_events_count:
                    break

            prob: float = 100.0 * switches_count / actual_cycles_count
            err: float = np.sqrt(prob * (100.0 - prob) / actual_cycles_count)
            print(f'for bias current set to {self.bias_current} nA, '
                  f'switching probability is {prob:.3f}% ± {err:.3f}%\n')
            self.results_queue.put((prob, err))
            if not self.stat_file.exists():
                self.stat_file.write_text('\t'.join((
                    'Set Bias Current [nA]',
                    'Temperature [mK]',
                    'Frequency [GHz]',
                    'Power [dBm]',
                    'Switch Probability [%]',
                    'Power [mW]',
                    'Actual Cycles Count',
                    'Probability Uncertainty [%]',
                )) + '\n', encoding='utf-8')
            with self.stat_file.open('at', encoding='utf-8') as f_out:
                f_out.write(f'{self.bias_current:.10f}'.rstrip('0').rstrip('.') + '\t' +
                            f'{self.temperature * 1000:.10f}'.rstrip('0').rstrip('.') + '\t' +
                            f'{self.frequency:.10f}'.rstrip('0').rstrip('.') + '\t' +
                            f'{self.power_dbm:.10f}'.rstrip('0').rstrip('.') + '\t' +
                            f'{prob:.10f}'.rstrip('0').rstrip('.') + '\t' +
                            f'{10.0 ** (0.1 * float(self.power_dbm)):.6e}\t'
                            f'{actual_cycles_count}\t'
                            f'{err}\n')
