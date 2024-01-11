# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Final, Literal, Sequence

import numpy as np
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.task import Task
from numpy.typing import NDArray

from hardware import (
    DIVIDER_RESISTANCE,
    adc_current,
    adc_sync,
    adc_voltage,
    dac_current,
    dac_sync,
    dac_synth_pulse,
    offsets,
)
from utils import error
from utils.connected_points import linear_segments, sine_segments
from utils.count import Count
from utils.filewriter import FileWriter
from utils.ni import measure_offsets
from utils.string_utils import format_float

__all__ = ["DetectMeasurement"]

fw: FileWriter = FileWriter()
fw.start()


class DetectMeasurement(Process):
    def __init__(
        self,
        results_queue: "Queue[tuple[float, float]]",
        state_queue: "Queue[tuple[int, int, int]]",
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
        max_reasonable_bias_error: float = np.inf,
        delay_between_cycles: float = 0.0,
        temperature: float = np.nan,
        adc_rate: float = np.nan,
    ) -> None:
        super(DetectMeasurement, self).__init__()

        self.results_queue: Queue[tuple[float, float]] = results_queue
        self.state_queue: Queue[tuple[int, int, int]] = state_queue
        self.good_to_go: SharedMemory = SharedMemory(name=good_to_go.name)

        self.voltage_gain: Final[float] = voltage_gain
        self.divider: Final[float] = current_divider
        self.r: Final[float] = resistance
        self.r_series: Final[float] = resistance_in_series

        self.bias_current: Final[float] = bias_current
        self.initial_biases: list[float] = list(initial_biases)
        self.setting_function: Final[str] = current_setting_function
        self.setting_time: Final[float] = setting_time

        self.pulse_duration: Final[float] = pulse_duration
        self.waiting_after_pulse: Final[float] = waiting_after_pulse
        self.power_dbm: Final[float] = power_dbm
        self.frequency: Final[float] = frequency

        self.trigger_voltage: float = trigger_voltage
        self.cycles_count: Final[int] = cycles_count
        self.max_switching_events_count: Final[int] = max_switching_events_count
        self.delay_between_cycles: Final[float] = delay_between_cycles
        self.max_reasonable_bias_error: Final[float] = max_reasonable_bias_error  # not used

        self.temperature: Final[float] = temperature

        self.adc_rate: float = adc_rate

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
            current_channel = task_dac.ao_channels.add_ao_voltage_chan(dac_current.name)
            task_dac.ao_channels.add_ao_voltage_chan(dac_synth_pulse.name)
            sync_channel = task_dac.ao_channels.add_ao_voltage_chan(dac_sync.name)

            if np.isnan(self.adc_rate):
                self.adc_rate = task_adc.timing.samp_clk_max_rate
            dac_rate: float = task_dac.timing.samp_clk_max_rate

            bias_current_steps_count: int = round(self.setting_time * dac_rate)
            pulse_duration_points_count: int = round(self.pulse_duration * dac_rate)
            waiting_after_pulse_points_count: int = round(self.waiting_after_pulse * dac_rate)

            samples_per_dac_channel: int = (
                2 * bias_current_steps_count + pulse_duration_points_count + waiting_after_pulse_points_count
            )
            if samples_per_dac_channel > task_dac.output_onboard_buffer_size:
                dac_rate /= samples_per_dac_channel / task_dac.output_onboard_buffer_size
                bias_current_steps_count = round(self.setting_time * dac_rate)
                pulse_duration_points_count = round(self.pulse_duration * dac_rate)
                waiting_after_pulse_points_count = round(self.waiting_after_pulse * dac_rate)
                samples_per_dac_channel = (
                    2 * bias_current_steps_count + pulse_duration_points_count + waiting_after_pulse_points_count
                )
            # The number of samples per channel to write multiplied by the number of channels in the task
            # cannot be an odd number for this device.
            spare_sample_count: int = (task_dac.number_of_channels * samples_per_dac_channel) % 2
            samples_per_dac_channel += spare_sample_count
            # If we get too many samples per channel again, we sacrifice the current steps
            while samples_per_dac_channel > task_dac.output_onboard_buffer_size:
                bias_current_steps_count -= 1
                samples_per_dac_channel = (
                    2 * bias_current_steps_count
                    + pulse_duration_points_count
                    + waiting_after_pulse_points_count
                    + spare_sample_count
                )

            trigger_trigger: float = 0.45 * sync_channel.ao_max
            trigger_sequence: NDArray[np.float64] = np.concatenate(
                (
                    np.zeros(bias_current_steps_count),
                    np.full(
                        pulse_duration_points_count + waiting_after_pulse_points_count,
                        2.0 * trigger_trigger,
                    ),
                    np.zeros(bias_current_steps_count + spare_sample_count),
                )
            )

            task_adc.timing.cfg_samp_clk_timing(
                rate=self.adc_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=1000,  # cannot set task_adc.input_onboard_buffer_size
            )
            task_dac.timing.cfg_samp_clk_timing(
                rate=dac_rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=samples_per_dac_channel,
            )

            adc_stream: AnalogMultiChannelReader = AnalogMultiChannelReader(task_adc.in_stream)

            def reading_task_callback(
                _task_idx: int,
                _event_type: int,
                num_samples: int,
                _callback_data: Any,
            ) -> Literal[0]:
                data: NDArray[np.float64] = np.empty((3, num_samples), dtype=np.float64)
                adc_stream.read_many_sample(data, num_samples)
                waiting: NDArray[np.bool_] = data[2] > trigger_trigger
                data[1] -= self.r_series / self.r * data[0] * self.voltage_gain
                not_switched: NDArray[np.bool_] = data[1] < self.trigger_voltage
                if np.any(waiting):
                    self.pulse_started = True
                    self.pulse_ended = not waiting[-1]
                    self.c.inc(np.count_nonzero(waiting & not_switched))
                    if self.c.loadable and not self.c.loaded and np.any(data[1] > self.trigger_voltage):
                        trig_arg: int = np.argwhere(data[1] > self.trigger_voltage).flat[0]
                        self.c.payload = (
                            data[0, trig_arg],
                            data[1, trig_arg],
                            int(self.c) / self.adc_rate,
                        )
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
            task_adc.register_every_n_samples_acquired_into_buffer_event(
                task_adc.timing.samp_quant_samp_per_chan,
                reading_task_callback,
            )

            task_adc.start()

            bias_current_amplitude: float = np.abs(float(self.bias_current) - self.initial_biases[-1])
            actual_bias_current_steps_count: int = round(
                bias_current_amplitude
                * 1e-9
                * self.r
                * self.divider
                / min(
                    ((current_channel.ao_max - current_channel.ao_min) / (2**current_channel.ao_resolution)),
                    bias_current_steps_count,
                )
            )
            actual_bias_current_step: float = bias_current_amplitude / (actual_bias_current_steps_count - 1)

            print(f"\nbias current is set to {self.bias_current} nA")
            print(f"number of current steps is {actual_bias_current_steps_count}")
            print(f"current step is {actual_bias_current_step:.4f} nA\n")
            if actual_bias_current_step > 4e-3 * bias_current_amplitude:
                error("current steps are too large")
                return

            # calculate the current sequence
            i_set: NDArray[np.float64]
            if self.setting_function.casefold() == "sine":
                i_set = (
                    np.concatenate(
                        (
                            sine_segments(
                                [self.initial_biases[-1], self.bias_current],
                                bias_current_steps_count,
                            ),
                            np.full(
                                pulse_duration_points_count + waiting_after_pulse_points_count,
                                self.bias_current,
                            ),
                            sine_segments(
                                [self.bias_current] + self.initial_biases,
                                bias_current_steps_count,
                            ),
                            [self.initial_biases[-1]] * spare_sample_count,
                        )
                    )
                    * 1e-9
                    * self.r
                    * self.divider
                )
            elif self.setting_function.casefold() == "linear":
                i_set = (
                    np.concatenate(
                        (
                            linear_segments(
                                [self.initial_biases[-1], self.bias_current],
                                bias_current_steps_count,
                            ),
                            np.full(
                                pulse_duration_points_count + waiting_after_pulse_points_count,
                                self.bias_current,
                            ),
                            linear_segments(
                                [self.bias_current] + self.initial_biases,
                                bias_current_steps_count,
                            ),
                            [self.initial_biases[-1]] * spare_sample_count,
                        )
                    )
                    * 1e-9
                    * self.r
                    * self.divider
                )
            else:
                raise ValueError("Unsupported current setting function:", self.setting_function)

            am_voltage_sequence: NDArray[np.float64] = np.concatenate(
                (
                    np.full(bias_current_steps_count, 0.0),
                    np.full(pulse_duration_points_count, 1.5),
                    np.full(waiting_after_pulse_points_count, 0.0),
                    np.full(bias_current_steps_count + spare_sample_count, 0.0),
                )
            )
            period_sequence: NDArray[np.float64] = np.row_stack(
                (
                    i_set * (1.0 + DIVIDER_RESISTANCE / self.r),
                    am_voltage_sequence,
                    trigger_sequence,
                )
            )

            # set initial state
            task_dac.write(
                np.row_stack(
                    (
                        np.full(
                            task_dac.timing.samp_quant_samp_per_chan,
                            (
                                min(
                                    0.0,
                                    self.initial_biases[-1] * 1e-9 * self.r * self.divider,
                                )
                                * (1.0 + DIVIDER_RESISTANCE / self.r)
                            ),
                        ),
                        np.zeros(task_dac.timing.samp_quant_samp_per_chan),
                        np.zeros(task_dac.timing.samp_quant_samp_per_chan),
                    )
                ),
                auto_start=True,
            )
            task_dac.wait_until_done()
            task_dac.stop()

            switches_count: int = 0
            actual_cycles_count: int = 0

            for cycle_index in range(self.cycles_count):
                while not self.good_to_go.buf[0] and not self.good_to_go.buf[127]:
                    time.sleep(1)
                if self.good_to_go.buf[127]:
                    break

                estimated_cycles_count: int
                if switches_count == 0:
                    estimated_cycles_count = self.cycles_count
                else:
                    estimated_cycles_count = min(
                        self.cycles_count,
                        round(self.max_switching_events_count * actual_cycles_count / switches_count),
                    )
                print(
                    f"{self.temperature * 1000:.1f} mK | {self.bias_current:.2f} nA | "
                    f"{self.frequency:.4f} GHz | {self.power_dbm:.2f} dBm | "
                    f"cycle {cycle_index + 1} out of {estimated_cycles_count}:",
                    end=" ",
                )

                while not self.c.loadable and not self.good_to_go.buf[127]:
                    time.sleep(0.01)
                if self.good_to_go.buf[127]:
                    print("user aborted")
                    break
                self.c.loaded = False
                self.pulse_started = False
                self.pulse_ended = False

                # set bias
                task_dac.write(period_sequence, auto_start=True)
                task_dac.wait_until_done()
                task_dac.stop()

                while not self.c.loadable or not self.pulse_ended and not self.good_to_go.buf[127]:
                    time.sleep(0.01)
                if self.good_to_go.buf[127]:
                    print("user aborted")
                    break
                self.pulse_started = False
                self.pulse_ended = False
                if self.c.loaded:
                    i, v, t = self.c.payload
                    self.c.loaded = False

                    v = (v - offsets[adc_voltage.name]) / self.voltage_gain
                    i = (i - v - offsets[adc_current.name]) / self.r

                    switches_count += 1
                    print("switching at" f" t = {t:.5f} s, {i * 1e9:.4f} nA, {v * 1e3:.4f} mV")
                    fw.write(self.data_file, "at", (i * 1e9, v * 1e3, t))
                else:
                    print("no switching events detected")
                    self.c.reset()

                self.state_queue.put((cycle_index, estimated_cycles_count, switches_count))

                actual_cycles_count = cycle_index + 1
                if switches_count >= self.max_switching_events_count:
                    break

                time.sleep(self.delay_between_cycles)

            prob: float = 100.0 * switches_count / actual_cycles_count if actual_cycles_count > 0 else np.nan
            err: float = np.sqrt(prob * (100.0 - prob) / actual_cycles_count) if actual_cycles_count > 0 else np.nan
            print(
                f"for bias current set to {self.bias_current} nA, "
                f"switching probability is {prob:.3f}% ± {err:.3f}%\n"
            )
            self.results_queue.put((prob, err))
            if not self.stat_file.exists():
                self.stat_file.write_text(
                    "\t".join(
                        (
                            "Set Bias Current [nA]",
                            "Temperature [mK]",
                            "Frequency [GHz]",
                            "Power [dBm]",
                            "Switch Probability [%]",
                            "Power [mW]",
                            "Actual Cycles Count",
                            "Probability Uncertainty [%]",
                            "Actual Temperature [mK]",
                        )
                    )
                    + "\n",
                    encoding="utf-8",
                )
            with self.stat_file.open("at", encoding="utf-8") as f_out:
                f_out.write(
                    "\t".join(
                        (
                            format_float(self.bias_current),
                            format_float(self.temperature * 1000),
                            format_float(self.frequency),
                            format_float(self.power_dbm),
                            format_float(prob),
                            f"{10.0 ** (0.1 * float(self.power_dbm)):.6e}",
                            str(actual_cycles_count),
                            str(err),
                            bytes(self.good_to_go.buf[1:65]).strip(b"\0").decode(),
                        )
                    )
                    + "\n"
                )
