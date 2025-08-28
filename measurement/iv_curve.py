import time
from datetime import datetime
from multiprocessing import Process
from multiprocessing.queues import Queue as QueueType
from typing import Final, Literal

import numpy as np
from nidaqmx.constants import WAIT_INFINITELY, AcquisitionType
from nidaqmx.errors import DaqError
from nidaqmx.stream_readers import AnalogMultiChannelReader, AnalogSingleChannelReader
from nidaqmx.stream_writers import AnalogSingleChannelWriter
from nidaqmx.task import Task
from nidaqmx.task.channels import AIChannel
from numpy.typing import NDArray

from hardware import (
    DIVIDER,
    DIVIDER_RESISTANCE,
    R_SERIES,
    VOLTAGE_GAIN,
    R,
    adc_current,
    adc_sync,
    adc_voltage,
    dac_current,
    dac_sync,
    offsets,
)
from utils.connected_points import linear_segment, parabolic_segment
from utils.ni import measure_offsets, zero_sources

__all__ = [
    "IVCurveMeasurement",
    "fast_iv_curve",
    "iv_curve",
    "iv_curve_2",
    "iv_curve_of_rate",
]


class IVCurveMeasurement(Process):
    def __init__(
        self,
        results_queue: QueueType[NDArray[np.float64]],
        min_current: float,
        max_current: float,
        current_rate: float,
        two_way: bool,
        ballast_resistance: float,
        voltage_gain: float,
        current_divider: float,
        adc_rate: float | None = None,
        resistance_in_series: float = 0.0,
        current_mode: str = "linear",
    ) -> None:
        super().__init__()

        self.results_queue: QueueType[NDArray[np.float64]] = results_queue

        self.min_current: float = min_current
        self.max_current: float = max_current
        self.current_rate: float = current_rate
        self.two_way: bool = two_way

        self.ballast_resistance: float = ballast_resistance
        self.voltage_gain: float = voltage_gain
        self.current_divider: float = current_divider
        self.resistance_in_series: float = resistance_in_series

        self.adc_rate: float | None = adc_rate

        self.current_mode: str = current_mode.casefold()

        self.pulse_started: bool = False
        self.pulse_ended: bool = False

    def run(self) -> None:
        measure_offsets()
        task_adc: Task
        task_dac: Task
        with Task() as task_dac:
            task_dac.ao_channels.add_ao_voltage_chan(dac_current.name)
            task_dac.write(self.min_current * self.ballast_resistance * self.current_divider)
            task_dac.wait_until_done()
            task_dac.stop()

        time.sleep(0.01)

        with Task() as task_adc, Task() as task_dac:
            adc_rate: float = task_adc.timing.samp_clk_max_rate if self.adc_rate is None else self.adc_rate
            c: AIChannel
            c = task_adc.ai_channels.add_ai_voltage_chan(adc_current.name)
            c.ai_enhanced_alias_rejection_enable = adc_rate is not None and adc_rate < 1000.0
            c = task_adc.ai_channels.add_ai_voltage_chan(adc_voltage.name)
            c.ai_enhanced_alias_rejection_enable = adc_rate is not None and adc_rate < 1000.0
            c = task_adc.ai_channels.add_ai_voltage_chan(adc_sync.name)
            c.ai_enhanced_alias_rejection_enable = adc_rate is not None and adc_rate < 1000.0
            task_dac.ao_channels.add_ao_voltage_chan(dac_current.name)
            sync_channel = task_dac.ao_channels.add_ao_voltage_chan(dac_sync.name)

            trigger_trigger: Final[float] = 0.45 * sync_channel.ao_max

            dac_rate: float = task_dac.timing.samp_clk_max_rate
            points: int = round(abs(self.max_current - self.min_current) / self.current_rate * dac_rate)
            samples_per_dac_channel: int = (2 if self.two_way else 1) * points + 2
            output_onbrd_buf_size: int = task_dac.out_stream.output_onbrd_buf_size
            if samples_per_dac_channel > output_onbrd_buf_size:
                dac_rate /= samples_per_dac_channel / output_onbrd_buf_size
                points = round(abs(self.max_current - self.min_current) / self.current_rate * dac_rate)
                samples_per_dac_channel = (2 if self.two_way else 1) * points + 2
            # If we get too many samples per channel again, we sacrifice the current steps
            while samples_per_dac_channel > output_onbrd_buf_size:
                points -= 1 if self.two_way else 2  # keep samples_per_dac_channel even
                samples_per_dac_channel = (2 if self.two_way else 1) * points + 2

            task_adc.timing.cfg_samp_clk_timing(
                rate=adc_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
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
                _callback_data: object,
            ) -> Literal[0]:
                data: NDArray[np.float64] = np.empty((3, num_samples), dtype=np.float64)
                adc_stream.read_many_sample(data, num_samples)
                waiting: NDArray[np.bool_] = data[2] > trigger_trigger
                if np.any(waiting):
                    self.pulse_started = True
                    self.pulse_ended = not waiting[-1]
                    data[0] -= offsets[adc_current.name]
                    data[1] -= offsets[adc_voltage.name]
                    data[0] /= self.ballast_resistance
                    data[1] /= self.voltage_gain
                    data[1] -= data[0] * self.resistance_in_series
                    data[0] -= data[1] / self.ballast_resistance
                    self.results_queue.put(data[0:2, waiting])
                elif self.pulse_started:
                    self.pulse_ended = True
                    self.pulse_started = False
                return 0

            # noinspection PyTypeChecker
            task_adc.register_every_n_samples_acquired_into_buffer_event(
                task_adc.timing.samp_quant_samp_per_chan,
                reading_task_callback,
            )

            # calculate the current sequence
            trigger_sequence: NDArray[np.float64] = np.concatenate(
                (
                    [0.0],
                    np.full((2 if self.two_way else 1) * points, 2.0 * trigger_trigger),
                    [0.0],
                ),
            )

            i_set: NDArray[np.float64]
            if self.current_mode == "linear":
                i_set = linear_segment(self.min_current, self.max_current, points)
            elif self.current_mode == "parabolic":
                i_set = parabolic_segment(self.min_current, self.max_current, points)
            else:
                raise ValueError("Invalid current mode")

            i_set *= self.current_divider * (DIVIDER_RESISTANCE + self.ballast_resistance)

            if self.two_way:
                i_set = np.concatenate(
                    (
                        i_set,
                        i_set[::-1],
                    ),
                )

            i_set = np.concatenate(([i_set[0]], i_set, [i_set[-1]]))

            task_adc.start()

            self.pulse_started = False
            self.pulse_ended = False
            task_dac.write(
                np.vstack((i_set, trigger_sequence)),
                auto_start=True,
                timeout=WAIT_INFINITELY,
            )
            task_dac.wait_until_done(timeout=WAIT_INFINITELY)
            task_dac.stop()

            while not self.pulse_ended:
                time.sleep(0.1)

            task_adc.stop()

        zero_sources()


def iv_curve(limits: tuple[float, float], points: int, two_way: bool = False) -> tuple[list[float], list[float]]:
    """Measure IV curve without actual current measurement."""
    task_adc: Task = Task()
    task_dac_current: Task = Task()
    task_adc.ai_channels.add_ai_voltage_chan(adc_voltage.name)
    task_dac_current.ao_channels.add_ao_voltage_chan(dac_current.name)
    task_adc.timing.cfg_samp_clk_timing(
        rate=task_adc.timing.samp_clk_max_rate,
        sample_mode=AcquisitionType.CONTINUOUS,
    )

    v: list[float] = []
    i: list[float] = []
    limits_min: float = min(limits)
    limits_ptp: float = max(limits) - limits_min

    adc_voltage_stream: AnalogSingleChannelReader = AnalogSingleChannelReader(task_adc.in_stream)

    def reading_task_callback(_task_idx: int, _event_type: int, num_samples: int, _callback_data: object) -> Literal[0]:
        # It may be wiser to read slightly more than num_samples here, to make sure one does not miss any sample,
        # see: https://documentation.help/NI-DAQmx-Key-Concepts/contCAcqGen.html
        _data = np.empty(num_samples)
        try:
            adc_voltage_stream.read_many_sample(_data, num_samples)
        except DaqError:
            pass
        else:
            task_adc.last_voltage = _data[-1]
        return 0

    task_adc.in_stream.auto_start = True
    # noinspection PyTypeChecker
    task_adc.register_every_n_samples_acquired_into_buffer_event(50, reading_task_callback)
    task_adc.last_voltage = np.nan
    task_adc.start()

    t0: float = datetime.now().timestamp()
    for k in range(points):
        current_i: float = limits_min + limits_ptp / points * k
        task_dac_current.write(current_i * R)
        v.append((task_adc.last_voltage - offsets[adc_voltage.name]) / VOLTAGE_GAIN)
        i.append(current_i)
    if two_way:
        for k in range(points):
            current_i: float = limits_min + limits_ptp / points * (points - k - 1)
            task_dac_current.write(current_i * R)
            v.append((task_adc.last_voltage - offsets[adc_voltage.name]) / VOLTAGE_GAIN)
            i.append(current_i)
    t1: float = datetime.now().timestamp()
    print(f"measurement took {t1 - t0} seconds ({(t1 - t0) / points / (2 if two_way else 1)} seconds per point)")
    task_adc.stop()

    task_adc.close()
    task_dac_current.close()

    return i, v


def iv_curve_2(limits: tuple[float, float], points: int, two_way: bool = False) -> tuple[list[float], list[float]]:
    """Measure IV curve with actual current measurement."""
    task_adc: Task = Task()
    task_dac_current: Task = Task()
    task_adc.ai_channels.add_ai_voltage_chan(adc_current.name)
    task_adc.ai_channels.add_ai_voltage_chan(adc_voltage.name)
    task_dac_current.ao_channels.add_ao_voltage_chan(dac_current.name)
    task_adc.timing.cfg_samp_clk_timing(
        rate=task_adc.timing.samp_clk_max_rate,
        sample_mode=AcquisitionType.CONTINUOUS,
    )

    v: list[float] = []
    i: list[float] = []
    limits_min: float = min(limits)
    limits_ptp: float = max(limits) - limits_min

    adc_voltage_stream: AnalogMultiChannelReader = AnalogMultiChannelReader(task_adc.in_stream)

    def reading_task_callback(_task_idx: int, _event_type: int, num_samples: int, _callback_data: object) -> Literal[0]:
        # It may be wiser to read slightly more than num_samples here, to make sure one does not miss any sample,
        # see: https://documentation.help/NI-DAQmx-Key-Concepts/contCAcqGen.html
        _data = np.empty((2, num_samples))
        try:
            adc_voltage_stream.read_many_sample(_data, num_samples)
        except DaqError:
            pass
        else:
            task_adc.last_current = _data[0, -1]
            task_adc.last_voltage = _data[1, -1]
        return 0

    # noinspection PyTypeChecker
    task_adc.register_every_n_samples_acquired_into_buffer_event(50, reading_task_callback)
    task_adc.last_voltage = np.nan
    task_adc.last_current = np.nan
    task_adc.start()

    t0: float = datetime.now().timestamp()
    for k in range(points):
        current_i: float = limits_min + limits_ptp / points * k
        task_dac_current.write(current_i * R)
        v.append((task_adc.last_voltage - offsets[adc_voltage.name]) / VOLTAGE_GAIN)
        i.append((task_adc.last_current - offsets[adc_current.name]) / R)
    if two_way:
        for k in range(points):
            current_i: float = limits_min + limits_ptp / points * (points - k - 1)
            task_dac_current.write(current_i * R)
            v.append((task_adc.last_voltage - offsets[adc_voltage.name]) / VOLTAGE_GAIN)
            i.append((task_adc.last_current - offsets[adc_current.name]) / R)
    t1: float = datetime.now().timestamp()
    print(f"measurement took {t1 - t0} seconds ({(t1 - t0) / points / (2 if two_way else 1)} seconds per point)")
    task_adc.stop()

    task_adc.close()
    task_dac_current.close()

    return i, v


def fast_iv_curve(
    limits: tuple[float, float],
    points: int,
    two_way: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Measure IV curve with actual current measurement."""
    task_adc: Task = Task()
    task_dac: Task = Task()
    task_adc.ai_channels.add_ai_voltage_chan(adc_current.name)
    task_adc.ai_channels.add_ai_voltage_chan(adc_voltage.name)
    current_channel = task_dac.ao_channels.add_ao_voltage_chan(dac_current.name)

    # rate: float = min(task_adc.timing.samp_clk_max_rate, task_dac.timing.samp_clk_max_rate)
    task_adc.timing.cfg_samp_clk_timing(
        rate=task_adc.timing.samp_clk_max_rate,
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=10000,
    )
    task_dac.timing.cfg_samp_clk_timing(
        rate=task_dac.timing.samp_clk_max_rate,
        # sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=(2 if two_way else 1) * points,
    )

    task_adc.i = np.empty(0)
    task_adc.v = np.empty(0)
    limits_min: float = min(limits)
    limits_max: float = max(limits)
    limits_ptp: float = limits_max - limits_min

    adc_stream: AnalogMultiChannelReader = AnalogMultiChannelReader(task_adc.in_stream)

    def reading_task_callback(_task_idx: int, _event_type: int, num_samples: int, _callback_data: object) -> Literal[0]:
        _data = np.empty((2, num_samples))
        adc_stream.read_many_sample(_data, num_samples)
        task_adc.i = np.concatenate((task_adc.i, _data[0]))
        task_adc.v = np.concatenate((task_adc.v, _data[1]))
        return 0

    # noinspection PyTypeChecker
    task_adc.register_every_n_samples_acquired_into_buffer_event(
        task_adc.timing.samp_quant_samp_per_chan,
        reading_task_callback,
    )
    i_set: NDArray[np.float64]
    if two_way:
        i_set = (
            np.concatenate(
                (
                    np.linspace(limits_min, limits_max, points, endpoint=True),
                    np.linspace(limits_max, limits_min, points, endpoint=True),
                ),
            )
            * R
        )
    else:
        i_set = np.linspace(limits_min, limits_max, points, endpoint=True, dtype=np.float64) * R
    i_set[i_set > current_channel.ao_max] = current_channel.ao_max
    i_set[i_set < current_channel.ao_min] = current_channel.ao_min

    task_dac.start()
    task_adc.start()
    t0: float = datetime.now().timestamp()
    written_points: int = task_dac.write(i_set)
    while written_points < i_set.size:
        written_points += task_dac.write(i_set[written_points:])
    t3: float = datetime.now().timestamp()
    print(
        task_dac.timing.samp_clk_rate,
        written_points / (t3 - t0 - 1 / 64) / (2 if two_way else 1),
    )
    print(limits_ptp / (t3 - t0) / (2 if two_way else 1))

    i: NDArray[np.float64]
    v: NDArray[np.float64]
    i = np.array(task_adc.i)
    v = np.array(task_adc.v)
    samples_left_unread: int = task_adc.in_stream.avail_samp_per_chan
    while samples_left_unread and datetime.now().timestamp() - t3 < 0.1:
        left_data: NDArray[np.float64] = task_adc.read(samples_left_unread)
        i = np.concatenate((i, left_data[0]))
        v = np.concatenate((v, left_data[1]))
        samples_left_unread = task_adc.in_stream.avail_samp_per_chan
    t1: float = datetime.now().timestamp()
    i = (i - offsets[adc_current.name]) / R
    v = (v - offsets[adc_voltage.name]) / VOLTAGE_GAIN

    task_adc.stop()
    task_dac.stop()
    task_adc.close()
    task_dac.close()

    if v.size:
        print(f"measurement took {t1 - t0} seconds ({v.size} points, {(t1 - t0) / v.size} seconds per point)")
    else:
        print(f"measurement took {t1 - t0} seconds")

    return i, v


def iv_curve_of_rate_bak(
    limits: tuple[float, float],
    current_rate: float,
    two_way: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Measure IV curve with actual current measurement."""
    task_adc: Task
    task_dac: Task
    with Task() as task_adc, Task() as task_dac:
        task_adc.ai_channels.add_ai_voltage_chan(adc_current.name)
        task_adc.ai_channels.add_ai_voltage_chan(adc_voltage.name)
        current_channel = task_dac.ao_channels.add_ao_voltage_chan(dac_current.name)
        limits_min: float = min(limits)
        limits_max: float = max(limits)
        limits_ptp: float = limits_max - limits_min

        points: int = round(limits_ptp / current_rate * task_dac.timing.samp_clk_max_rate)
        # points / task_dac.timing.samp_clk_max_rate == limits_ptp / current_rate
        task_adc.timing.cfg_samp_clk_timing(
            rate=task_adc.timing.samp_clk_max_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=10000,
        )
        task_dac.timing.cfg_samp_clk_timing(
            rate=task_dac.timing.samp_clk_max_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=(2 if two_way else 1) * points,
        )

        task_adc.i = np.empty(0)
        task_adc.v = np.empty(0)

        adc_stream: AnalogMultiChannelReader = AnalogMultiChannelReader(task_adc.in_stream)
        dac_stream: AnalogSingleChannelWriter = AnalogSingleChannelWriter(task_dac.out_stream, auto_start=True)

        def reading_task_callback(
            _task_idx: int, _event_type: int, num_samples: int, _callback_data: object
        ) -> Literal[0]:
            _data = np.empty((2, num_samples))
            adc_stream.read_many_sample(_data, num_samples)
            task_adc.i = np.concatenate((task_adc.i, _data[0]))
            task_adc.v = np.concatenate((task_adc.v, _data[1]))
            return 0

        # noinspection PyTypeChecker
        task_adc.register_every_n_samples_acquired_into_buffer_event(
            task_adc.timing.samp_quant_samp_per_chan,
            reading_task_callback,
        )

        # calculating the current sequence
        i_set: NDArray[np.float64]
        if two_way:
            i_set = (
                np.concatenate(
                    (
                        np.linspace(limits_min, limits_max, points, endpoint=True),
                        np.linspace(limits_max, limits_min, points, endpoint=True),
                    ),
                )
                * R
            )
        else:
            i_set = np.linspace(limits_min, limits_max, points, endpoint=True, dtype=np.float64) * R
        i_set[i_set > current_channel.ao_max] = current_channel.ao_max
        i_set[i_set < current_channel.ao_min] = current_channel.ao_min

        task_dac.start()
        task_adc.start()

        t0: float = datetime.now().timestamp()
        written_points: int = dac_stream.write_many_sample(i_set)
        while written_points < i_set.size:
            written_points += dac_stream.write_many_sample(i_set[written_points:])
        t3: float = datetime.now().timestamp()
        # print(task_dac.timing.samp_clk_rate,
        #       written_points / (t3 - t0 - 1 / 64) * (2 if two_way else 1),
        #       written_points / (t3 - t0 + 1 / 64) * (2 if two_way else 1),
        #       )
        # print(limits_ptp / (t3 - t0 + 1 / 64) * (2 if two_way else 1),
        #       limits_ptp / (t3 - t0 - 1 / 64) * (2 if two_way else 1))

        i: NDArray[np.float64]
        v: NDArray[np.float64]
        i = np.array(task_adc.i)
        v = np.array(task_adc.v)
        samples_left_unread: int = task_adc.in_stream.avail_samp_per_chan
        while samples_left_unread and datetime.now().timestamp() - t3 < 0.2:
            left_data: NDArray[np.float64] = task_adc.read(samples_left_unread)
            i = np.concatenate((i, left_data[0]))
            v = np.concatenate((v, left_data[1]))
            samples_left_unread = task_adc.in_stream.avail_samp_per_chan
        t1: float = datetime.now().timestamp()
        i = (i - offsets[adc_current.name]) / R
        v = (v - offsets[adc_voltage.name]) / VOLTAGE_GAIN

        task_adc.stop()
        task_dac.stop()

    if v.size:
        print(f"measurement took {t1 - t0} seconds ({v.size} points, {(t1 - t0) / v.size} seconds per point)")
    else:
        print(f"measurement took {t1 - t0} seconds")

    return i, v


def iv_curve_of_rate(
    limits: tuple[float, float],
    current_rate: float,
    two_way: bool = False,
    *,
    ballast_resistance: float = R,
    current_divider: float = DIVIDER,
    voltage_gain: float = VOLTAGE_GAIN,
    resistance_in_series: float = R_SERIES,
    current_mode: str = "linear",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Measure IV curve with actual current measurement."""
    min_current: float = min(limits)
    max_current: float = max(limits)
    limits_ptp: float = max_current - min_current
    current_mode = current_mode.casefold()

    task_adc: Task
    task_dac: Task
    with Task() as task_dac:
        task_dac.ao_channels.add_ao_voltage_chan(dac_current.name)
        task_dac.write(min_current * ballast_resistance * current_divider)

    time.sleep(0.01)

    with Task() as task_adc, Task() as task_dac:
        task_adc.ai_channels.add_ai_voltage_chan(adc_current.name)
        task_adc.ai_channels.add_ai_voltage_chan(adc_voltage.name)
        task_adc.ai_channels.add_ai_voltage_chan(adc_sync.name)
        task_dac.ao_channels.add_ao_voltage_chan(dac_current.name)
        sync_channel = task_dac.ao_channels.add_ao_voltage_chan(dac_sync.name)

        trigger_trigger: Final[float] = 0.45 * sync_channel.ao_max

        points: int = round(limits_ptp / current_rate * task_dac.timing.samp_clk_max_rate)
        # points / task_dac.timing.samp_clk_max_rate == limits_ptp / current_rate
        task_adc.timing.cfg_samp_clk_timing(
            rate=task_adc.timing.samp_clk_max_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=10000,
        )
        task_dac.timing.cfg_samp_clk_timing(
            rate=task_dac.timing.samp_clk_max_rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=(2 if two_way else 1) * points + 2,
        )

        i: NDArray[np.float64] = np.empty(0)
        v: NDArray[np.float64] = np.empty(0)

        adc_stream: AnalogMultiChannelReader = AnalogMultiChannelReader(task_adc.in_stream)

        no_data: bool = False

        def reading_task_callback(
            _task_idx: int,
            _event_type: int,
            num_samples: int,
            _callback_data: object,
        ) -> Literal[0]:
            nonlocal no_data, adc_stream, i, v
            data: NDArray[np.float64] = np.empty((3, num_samples))
            adc_stream.read_many_sample(data, num_samples)
            data = data[0:2, data[2] > trigger_trigger]
            i = np.concatenate((i, data[0]))
            v = np.concatenate((v, data[1]))
            no_data = bool(data.size == 0)
            return 0

        # noinspection PyTypeChecker
        task_adc.register_every_n_samples_acquired_into_buffer_event(
            task_adc.timing.samp_quant_samp_per_chan,
            reading_task_callback,
        )

        # calculating the current sequence
        i_set: NDArray[np.float64]
        trigger_on_sequence: NDArray[np.float64]
        if current_mode == "linear":
            if two_way:
                i_set = (
                    np.concatenate(
                        (
                            [min_current],
                            np.linspace(min_current, max_current, points, endpoint=True),
                            np.linspace(max_current, min_current, points, endpoint=True),
                            [min_current],
                        ),
                    )
                    * ballast_resistance
                    * current_divider
                )
                trigger_on_sequence = np.concatenate(([0.0], np.full(2 * points, 2.0 * trigger_trigger), [0.0]))
            else:
                i_set = (
                    np.concatenate(
                        (
                            [min_current],
                            np.linspace(min_current, max_current, points, endpoint=True),
                            [max_current],
                        ),
                    )
                    * ballast_resistance
                    * current_divider
                )
                trigger_on_sequence = np.concatenate(([0.0], np.full(points, 2.0 * trigger_trigger), [0.0]))
        elif current_mode == "parabolic":
            if min_current <= 0 and max_current <= 0:
                i_set = -np.square(
                    np.linspace(
                        np.sqrt(-min_current),
                        np.sqrt(-max_current),
                        points,
                        endpoint=True,
                    ),
                )
            elif min_current <= 0 <= max_current:
                i_set = np.concatenate(
                    (
                        -np.square(
                            np.linspace(
                                np.sqrt(-min_current),
                                0.0,
                                round(points * abs(min_current / limits_ptp)),
                                endpoint=True,
                            ),
                        ),
                        np.square(
                            np.linspace(
                                0.0,
                                np.sqrt(max_current),
                                round(points * abs(max_current / limits_ptp)),
                                endpoint=True,
                            ),
                        ),
                    ),
                )
            elif min_current >= 0 >= max_current:
                i_set = np.concatenate(
                    (
                        np.square(
                            np.linspace(
                                np.sqrt(min_current),
                                0.0,
                                round(points * abs(min_current / limits_ptp)),
                                endpoint=True,
                            ),
                        ),
                        -np.square(
                            np.linspace(
                                0.0,
                                np.sqrt(-max_current),
                                round(points * abs(max_current / limits_ptp)),
                                endpoint=True,
                            ),
                        ),
                    ),
                )
            else:
                i_set = np.square(
                    np.linspace(
                        np.sqrt(min_current),
                        np.sqrt(max_current),
                        points,
                        endpoint=True,
                    ),
                )
            if two_way:
                i_set = (
                    np.concatenate(([min_current], i_set, i_set[::-1], [min_current]))
                    * ballast_resistance
                    * current_divider
                )
                trigger_on_sequence = np.concatenate(([0.0], np.full(2 * points, 2.0 * trigger_trigger), [0.0]))
            else:
                i_set = np.concatenate(([min_current], i_set, [max_current])) * ballast_resistance * current_divider
                trigger_on_sequence = np.concatenate(([0.0], np.full(points, 2.0 * trigger_trigger), [0.0]))
        else:
            raise ValueError("Invalid current mode")

        task_adc.start()

        # t0: float = datetime.now().timestamp()
        task_dac.write(
            np.vstack((i_set, trigger_on_sequence)),
            auto_start=True,
            timeout=task_dac.timing.samp_quant_samp_per_chan / task_dac.timing.samp_clk_rate * 2.0,
        )
        task_dac.wait_until_done(timeout=task_dac.timing.samp_quant_samp_per_chan / task_dac.timing.samp_clk_rate * 2.0)
        task_dac.stop()

        while not no_data:
            time.sleep(0.01)

        # t1: float = datetime.now().timestamp()
        v = (v - offsets[adc_voltage.name]) / voltage_gain
        i = (i - offsets[adc_current.name] - v) / ballast_resistance
        v -= i * resistance_in_series
        task_adc.stop()

    # if v.size:
    #     print(f' measurement took {t1 - t0} seconds ({v.size} points, {(t1 - t0) / v.size} seconds per point)')
    # else:
    #     print(f' measurement took {t1 - t0} seconds')

    return i, v
