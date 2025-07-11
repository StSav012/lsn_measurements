import time
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from multiprocessing import Event, Process, Value
from multiprocessing.queues import Queue as QueueType
from pathlib import Path
from typing import Any, Final, Literal, cast

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
    offsets,
)
from utils import error
from utils.connected_points import half_sine_segments, linear_segments, quarter_sine_segments
from utils.count import Count
from utils.filewriter import FileWriter
from utils.ni import measure_offsets
from utils.string_utils import format_float

fw: FileWriter = FileWriter()
fw.start()

__all__ = ["LifetimeMeasurement"]


class LifetimeMeasurement(Process):
    def __init__(
        self,
        results_queue: QueueType[tuple[float, float, float]],
        state_queue: QueueType[tuple[int, timedelta]],
        good_to_go: Event,
        user_aborted: Event,
        actual_temperature: Value,
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
        temperature: float = np.nan,
        adc_rate: float = np.nan,
    ) -> None:
        super().__init__()

        self.results_queue: QueueType[tuple[float, float, float]] = results_queue
        self.state_queue: QueueType[tuple[int, timedelta]] = state_queue
        self.good_to_go: Event = good_to_go
        self.user_aborted: Event = user_aborted
        self.actual_temperature: Value = actual_temperature

        self.gain: Final[float] = voltage_gain
        self.divider: Final[float] = current_divider
        self.r: Final[float] = resistance
        self.r_series: Final[float] = resistance_in_series

        self.bias_current: Final[float] = bias_current
        self.initial_biases: list[float] = list(initial_biases)
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

        self.adc_rate: float = adc_rate

        self.stat_file: Path = stat_file
        self.data_file: Path = data_file

        self.c: Count = Count()

    def run(self) -> None:
        if self.data_file.exists():
            fw.write(self.data_file, "at", [])
        else:
            fw.write(
                self.data_file,
                "wt",
                [
                    "Frequency [GHz]",
                    "Set Bias Current [nA]",
                    "Lifetime [s]",
                    "Switch Bias Current [nA]",
                    "Switch Voltage [mV]",
                    "Measurement Duration [s]",
                    "Temperature [mK]",
                ],
            )

        measure_offsets()
        self.trigger_voltage += offsets[adc_voltage.name]

        task_adc: Task
        task_dac: Task
        with Task() as task_adc, Task() as task_dac:
            task_adc.ai_channels.add_ai_voltage_chan(adc_current.name)
            task_adc.ai_channels.add_ai_voltage_chan(adc_voltage.name)
            task_adc.ai_channels.add_ai_voltage_chan(adc_sync.name)
            current_channel = task_dac.ao_channels.add_ao_voltage_chan(dac_current.name)
            sync_channel = task_dac.ao_channels.add_ao_voltage_chan(dac_sync.name)

            if np.isnan(self.adc_rate):
                self.adc_rate = task_adc.timing.samp_clk_max_rate
            dac_rate: float = task_dac.timing.samp_clk_max_rate
            bias_current_steps_count: int = round(self.setting_time * dac_rate)

            input_onboard_buffer_size: int = task_adc.in_stream.input_onbrd_buf_size
            if bias_current_steps_count > input_onboard_buffer_size:
                dac_rate /= bias_current_steps_count / input_onboard_buffer_size
                bias_current_steps_count = round(self.setting_time * dac_rate)
            # If we get too many samples per channel again, we sacrifice the current steps
            while bias_current_steps_count > input_onboard_buffer_size:
                bias_current_steps_count -= 1

            trigger_trigger: float = 0.45 * sync_channel.ao_max
            trigger_on_sequence: NDArray[np.float64] = np.zeros(bias_current_steps_count, dtype=np.float64)
            trigger_on_sequence[-1] = 2.0 * trigger_trigger
            trigger_off_sequence: NDArray[np.float64] = np.zeros(bias_current_steps_count, dtype=np.float64)

            task_adc.timing.cfg_samp_clk_timing(
                rate=self.adc_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=1000,
            )
            task_dac.timing.cfg_samp_clk_timing(
                rate=dac_rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=bias_current_steps_count,
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
                data[1] -= self.r_series / self.r * data[0] * self.gain
                not_switched: NDArray[np.bool_] = data[1, waiting] < self.trigger_voltage
                if np.any(waiting):
                    if self.c.loadable and not self.c.loaded:
                        this_time_not_switched: int = np.count_nonzero(not_switched)
                        self.c.inc(this_time_not_switched)
                        if not_switched.size > this_time_not_switched:
                            trig_arg: int = np.argwhere(data[1] > self.trigger_voltage).flat[0]
                            self.c.payload = (
                                data[0, trig_arg],
                                data[1, trig_arg],
                                int(self.c) / self.adc_rate,
                            )
                            self.c.loaded = True
                            self.c.reset()
                else:
                    self.c.loadable = np.any(data[1] < 0.5 * self.trigger_voltage)
                return 0

            # noinspection PyTypeChecker
            task_adc.register_every_n_samples_acquired_into_buffer_event(
                task_adc.timing.samp_quant_samp_per_chan,
                reading_task_callback,
            )

            task_adc.start()

            task_dac.write(
                np.vstack(
                    (
                        np.full(
                            bias_current_steps_count,
                            self.initial_biases[-1] * 1e-9 * self.divider * (DIVIDER_RESISTANCE + self.r),
                        ),
                        trigger_off_sequence,
                    ),
                ),
                auto_start=True,
            )
            task_dac.wait_until_done()
            task_dac.stop()

            measurement_start_time: datetime = datetime.now()

            bias_current_amplitude: float = np.abs(float(self.bias_current) - self.initial_biases[-1])
            actual_bias_current_steps_count: int = min(
                round(
                    bias_current_amplitude
                    * 1e-9
                    * self.r
                    * self.divider
                    / ((current_channel.ao_max - current_channel.ao_min) / (2**current_channel.ao_resolution)),
                ),
                bias_current_steps_count,
            )
            actual_bias_current_step: float = bias_current_amplitude / (actual_bias_current_steps_count - 1)

            print(f"\nbias current is set to {self.bias_current} nA")
            print(f"number of current steps is {actual_bias_current_steps_count}")
            print(f"current step is {actual_bias_current_step:.4f} nA\n")
            if actual_bias_current_step > 4e-3 * bias_current_amplitude:
                error("current steps are too large")
                return

            # calculate the current sequence
            reset_function: Callable[[Sequence[float], int], NDArray[np.float64]]
            match self.reset_function.casefold():
                case "sine" | "half sine":
                    reset_function = half_sine_segments
                case "quarter sine":
                    reset_function = quarter_sine_segments
                case "linear":
                    reset_function = linear_segments
                case _:
                    raise ValueError("Unsupported current setting function:", self.reset_function)

            initial_i_set: NDArray[np.float64] = (
                reset_function(
                    [0.0, float(self.bias_current)],
                    bias_current_steps_count,
                )
                * self.r
                * self.divider
            )
            i_set: NDArray[np.float64] = (
                reset_function(
                    [self.initial_biases[-1], float(self.bias_current)],
                    bias_current_steps_count,
                )
                * self.r
                * self.divider
            )
            i_unset: NDArray[np.float64] = (
                reset_function(
                    [float(self.bias_current)] + self.initial_biases,
                    bias_current_steps_count,
                )
                * self.r
                * self.divider
            )

            initial_i_set = np.vstack(
                (
                    initial_i_set * (1.0 + DIVIDER_RESISTANCE / self.r) * 1e-9,
                    trigger_on_sequence,
                ),
            )
            i_set = np.vstack(
                (
                    i_set * (1.0 + DIVIDER_RESISTANCE / self.r) * 1e-9,
                    trigger_on_sequence,
                ),
            )
            i_unset = np.vstack(
                (
                    i_unset * (1.0 + DIVIDER_RESISTANCE / self.r) * 1e-9,
                    trigger_off_sequence,
                ),
            )

            task_dac.write(initial_i_set, auto_start=True)
            task_dac.wait_until_done()
            task_dac.stop()
            task_dac.write(i_unset, auto_start=True)
            task_dac.wait_until_done()
            task_dac.stop()

            switching_time: NDArray[np.float64] = np.full(self.cycles_count, np.nan, dtype=np.float64)
            set_bias_current: NDArray[np.float64] = np.full(self.cycles_count, np.nan, dtype=np.float64)

            for cycle_index in range(self.cycles_count):
                if not self.good_to_go.is_set():
                    while not self.good_to_go.wait(0.1) and not self.user_aborted.wait(0.1):
                        ...
                    if not self.user_aborted.is_set():
                        task_dac.write(i_set, auto_start=True)
                        task_dac.wait_until_done()
                        task_dac.stop()
                        task_dac.write(i_unset, auto_start=True)
                        task_dac.wait_until_done()
                        task_dac.stop()

                if self.user_aborted.is_set():
                    break

                # set initial state
                task_dac.write(
                    np.vstack(
                        (
                            np.full(
                                bias_current_steps_count,
                                (
                                    min(
                                        0.0,
                                        self.initial_biases[-1] * 1e-9 * self.r * self.divider,
                                    )
                                    * (1.0 + DIVIDER_RESISTANCE / self.r)
                                ),
                            ),
                            np.zeros(bias_current_steps_count),
                        ),
                    ),
                    auto_start=True,
                )
                task_dac.wait_until_done()
                task_dac.stop()
                while not self.c.loadable and not self.user_aborted.wait(0.01):
                    ...
                if self.user_aborted.is_set():
                    break
                self.c.loaded = False

                print(
                    datetime.now(),
                    f"cycle {cycle_index + 1} out of {self.cycles_count}:",
                    end=" ",
                )

                # set bias
                task_dac.write(i_set, auto_start=True)
                task_dac.wait_until_done()
                task_dac.stop()

                t0: datetime = datetime.now()
                t1: datetime = datetime.now()

                while t1 - t0 <= self.max_waiting_time and not self.c.loaded and not self.user_aborted.wait(0.01):
                    self.state_queue.put((cycle_index, t1 - t0))
                    t1 = datetime.now()

                if self.c.loaded:
                    self.c.loadable = False
                    i, v, t = self.c.payload
                    v = (v - offsets[adc_voltage.name]) / self.gain
                    i = (i - v - offsets[adc_current.name]) / self.r

                    set_bias_current[cycle_index] = i
                    switching_time[cycle_index] = t
                    print(f"switching at t = {t:.5f} s, {i * 1e9} nA, {v * 1e3} mV")
                    self.state_queue.put((cycle_index, timedelta(seconds=t)))
                    fw.write(
                        self.data_file,
                        "at",
                        [
                            self.frequency,
                            self.bias_current,
                            t,
                            i * 1e9,
                            v * 1e3,
                            (datetime.now() - measurement_start_time).total_seconds(),
                            self.actual_temperature.value,
                        ],
                    )

                    self.c.loaded = False
                elif self.user_aborted.is_set():
                    print("user aborted")
                else:
                    print("no switching events detected")
                    self.c.reset()
                    if not self.ignore_never_switched:
                        i, v = np.nan, np.nan
                        switching_time[cycle_index] = self.max_waiting_time.total_seconds()
                        self.state_queue.put((cycle_index, self.max_waiting_time))
                        fw.write(
                            self.data_file,
                            "at",
                            [
                                self.frequency,
                                self.bias_current,
                                self.max_waiting_time.total_seconds(),
                                i * 1e9,
                                v * 1e3,
                                (datetime.now() - measurement_start_time).total_seconds(),
                                self.actual_temperature.value,
                            ],
                        )

                task_dac.write(i_unset, auto_start=True)
                task_dac.wait_until_done()
                task_dac.stop()

                time.sleep(self.delay_between_cycles)

            if np.count_nonzero(~np.isnan(switching_time)) > 1:
                print(
                    f"for bias current set to {self.bias_current} nA, "
                    f"mean switching time is {np.nanmean(switching_time)} s "
                    f"± {np.nanstd(switching_time)} s",
                )
                # set_bias_current = (set_bias_current - offsets[adc_current.name]) / r

                # noinspection PyTypeChecker
                median_bias_current: float = np.nanmedian(set_bias_current)
                min_reasonable_bias_current: float = median_bias_current * (1.0 - self.max_reasonable_bias_error)
                max_reasonable_bias_current: float = median_bias_current * (1.0 + self.max_reasonable_bias_error)
                reasonable: NDArray[np.bool_] = (set_bias_current >= min_reasonable_bias_current) & (
                    set_bias_current <= max_reasonable_bias_current
                )
                set_bias_current_reasonable: NDArray[np.float64] = set_bias_current[reasonable] * 1e9
                mean_set_bias_current_reasonable: float | np.float64 | NDArray[np.float64]
                set_bias_current_reasonable_std: float | np.float64 | NDArray[np.float64]
                if set_bias_current_reasonable.size:
                    mean_set_bias_current_reasonable = np.nanmean(set_bias_current_reasonable)
                    set_bias_current_reasonable_std = np.nanstd(set_bias_current_reasonable)
                else:
                    mean_set_bias_current_reasonable = np.nan
                    set_bias_current_reasonable_std = np.nan

                switching_time_reasonable: NDArray[np.float64] = switching_time[reasonable]
                mean_switching_time_reasonable: float | np.float64 | NDArray[np.float64]
                switching_time_reasonable_std: float | np.float64 | NDArray[np.float64]
                if switching_time_reasonable.size:
                    mean_switching_time_reasonable = np.nanmean(switching_time_reasonable)
                    switching_time_reasonable_std = np.nanstd(switching_time_reasonable)
                else:
                    mean_switching_time_reasonable = np.nan
                    switching_time_reasonable_std = np.nan

                non_zero: NDArray[np.bool_] = switching_time_reasonable > 0.0
                switching_time_rnz: NDArray[np.float64] = switching_time_reasonable[non_zero]
                mean_switching_time_rnz: float | np.float64 | NDArray[np.float64]
                switching_time_rnz_std: float | np.float64 | NDArray[np.float64]
                if switching_time_rnz.size:
                    mean_switching_time_rnz = np.nanmean(switching_time_rnz)
                    switching_time_rnz_std = np.nanstd(switching_time_rnz)
                else:
                    mean_switching_time_rnz = np.nan
                    switching_time_rnz_std = np.nan
                if not self.stat_file.exists():
                    self.stat_file.write_text(
                        "\t".join(
                            (
                                "Frequency [GHz]",
                                "Set Bias Current [nA]",
                                "Mean Bias Current [nA]",
                                "Bias Current StD [nA]",
                                "τ₀ [s]",
                                "σ₀ [s]",
                                "τ [s]",
                                "σ [s]",
                                "τ₀/σ₀",
                                "τ/σ",
                                "Temperature [mK]",
                                "1/τ₀ [1/s]",
                                "1/τ [1/s]",
                                "Cycles",
                            ),
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                with self.stat_file.open("at", encoding="utf-8") as f_out:
                    f_out.write(
                        "\t".join(
                            (
                                # Frequency [GHz]
                                format_float(self.frequency),
                                # Set Bias Current [nA]
                                format_float(self.bias_current),
                                # Mean Bias Current [nA]
                                format_float(mean_set_bias_current_reasonable),
                                # Bias Current StD [nA]
                                format_float(set_bias_current_reasonable_std),
                                # τ₀ [s]
                                format_float(mean_switching_time_reasonable),
                                # σ₀ [s]
                                format_float(switching_time_reasonable_std),
                                # τ [s]
                                format_float(mean_switching_time_rnz),
                                # σ [s]
                                format_float(switching_time_rnz_std),
                                # τ₀/σ₀
                                (
                                    format_float(mean_switching_time_reasonable / switching_time_reasonable_std)
                                    if switching_time_reasonable_std
                                    else "nan"
                                ),
                                # τ/σ
                                (
                                    format_float(mean_switching_time_rnz / switching_time_rnz_std)
                                    if switching_time_rnz_std
                                    else "nan"
                                ),
                                # Temperature [mK]
                                format_float(self.actual_temperature.value),
                                # 1/τ₀ [1/s]
                                (
                                    format_float(1.0 / mean_switching_time_reasonable)
                                    if mean_switching_time_reasonable
                                    else "nan"
                                ),
                                # 1/τ [1/s]
                                format_float(1.0 / mean_switching_time_rnz) if mean_switching_time_rnz else "nan",
                                # Cycles
                                str(np.count_nonzero(~np.isnan(switching_time))),
                            ),
                        )
                        + "\n",
                    )
                self.results_queue.put(
                    (
                        cast("float", mean_set_bias_current_reasonable),
                        cast("float", mean_switching_time_reasonable),
                        cast("float", mean_switching_time_rnz),
                    ),
                )
            elif self.user_aborted.is_set():
                print("user aborted")
            else:
                print(f"no switching event detected for bias current set to {self.bias_current} nA")
