import sys
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from typing import Final, cast

import numpy as np
from nidaqmx.constants import WAIT_INFINITELY, AcquisitionType
from nidaqmx.system.physical_channel import PhysicalChannel
from nidaqmx.task import Task
from nidaqmx.task.channels import AIChannel
from numpy.typing import NDArray
from scipy import signal

from hardware import adc_current, adc_voltage, device_adc, device_dac, offsets

__all__ = [
    "measure_noise_fft",
    "measure_noise_trend",
    "measure_noise_welch",
    "measure_noise_welch_iter",
    "measure_noise_welch_iter_multiple",
    "measure_offsets",
    "zero_sources",
]

_RESET_ADC_DEFAULT: Final[bool] = device_adc.name != device_dac.name


def zero_sources(*, reset_dac: bool = True, exceptions: Sequence[PhysicalChannel] = ()) -> None:
    if reset_dac:
        device_dac.reset_device()
    task_dac: Task
    with Task() as task_dac:
        channel: PhysicalChannel
        for channel in device_dac.ao_physical_chans:
            if channel not in exceptions:
                task_dac.ao_channels.add_ao_voltage_chan(channel.name)
        task_dac.write([0.0] * task_dac.number_of_channels)
        task_dac.wait_until_done()


def measure_offsets(
    duration: float = 0.04,
    *,
    do_zero_sources: bool = True,
    reset_adc: bool = _RESET_ADC_DEFAULT,
) -> None:
    if reset_adc:
        device_adc.reset_device()
    task_adc: Task
    with Task() as task_adc:
        channel: PhysicalChannel
        for channel in device_adc.ai_physical_chans:
            task_adc.ai_channels.add_ai_voltage_chan(channel.name)
        count: int = round(duration * task_adc.timing.samp_clk_max_rate)
        task_adc.timing.cfg_samp_clk_timing(
            rate=task_adc.timing.samp_clk_max_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
        )
        input_onboard_buffer_size: int = task_adc.in_stream.input_onbrd_buf_size
        if do_zero_sources:
            zero_sources()
        task_adc.start()
        data_chunks: deque[NDArray[np.float64]] = deque()
        for _ in range(input_onboard_buffer_size // count):
            data_chunks.append(task_adc.read(input_onboard_buffer_size, timeout=WAIT_INFINITELY))
        data_chunks.append(task_adc.read(input_onboard_buffer_size % count, timeout=WAIT_INFINITELY))
        task_adc.stop()
        data: NDArray[np.float64] = np.concatenate(data_chunks)
        index: int
        for index, channel in enumerate(device_adc.ai_physical_chans):
            offsets[channel.name] = cast("float", np.mean(data[index]))


def measure_noise_fft(
    length: int,
    *,
    rate: float | None = None,
    reset_adc: bool = _RESET_ADC_DEFAULT,
) -> tuple[
    NDArray[np.float64],
    tuple[NDArray[np.float64], str],
    tuple[NDArray[np.float64], str],
]:
    if reset_adc:
        device_adc.reset_device()
    task_adc: Task
    with Task() as task_adc:
        task_adc.ai_channels.add_ai_voltage_chan(adc_current.name)
        task_adc.ai_channels.add_ai_voltage_chan(adc_voltage.name)
        if rate is None:
            rate = task_adc.timing.samp_clk_max_rate
        task_adc.timing.cfg_samp_clk_timing(
            rate=rate,
            sample_mode=AcquisitionType.FINITE,  # don't use continuous
            samps_per_chan=length,
        )
        zero_sources()
        data = np.asarray(task_adc.read(length))

        task_adc.close()

        freq: NDArray[np.float64] = np.fft.fftfreq(data.shape[1], 1.0 / rate)

        return (
            np.abs(freq),
            (np.square(np.abs(np.fft.fft(data[0], norm="ortho"))), adc_current.name),
            (np.square(np.abs(np.fft.fft(data[1], norm="ortho"))), adc_voltage.name),
        )


def measure_noise_welch(
    channel: str,
    resolution: float,
    rate: float | None = None,
    *,
    averaging: int = 1,
    averaging_shift: float = 0.0,
    reset_adc: bool = _RESET_ADC_DEFAULT,
    progress: str = "",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if averaging < 1:
        raise ValueError("Averaging must be a positive number")

    if reset_adc:
        device_adc.reset_device()
    task_adc: Task
    with Task() as task_adc:
        task_adc.ai_channels.add_ai_voltage_chan(channel)
        if rate is None:
            rate = task_adc.timing.samp_clk_max_rate

        averaging_step: int = round(rate * averaging_shift)
        if averaging != 1 and averaging_step < 1:
            raise ValueError(f"Averaging shift must be greater than {0.5 / rate}")

        length: int = round(rate / resolution) + (averaging - 1) * averaging_step
        task_adc.timing.cfg_samp_clk_timing(rate=rate, sample_mode=AcquisitionType.CONTINUOUS)
        task_adc.start()
        if progress:
            sys.stdout.write(progress)
        data: NDArray[np.float64] = np.asarray(task_adc.read(length, timeout=WAIT_INFINITELY), dtype=np.float64)
        if progress:
            sys.stdout.write(progress)
        task_adc.stop()

    freq: NDArray[np.float64]
    pn_xx: NDArray[np.float64]
    if averaging == 1:
        freq, pn_xx = signal.welch(data, fs=rate, nperseg=data.size)
    else:
        freq, pn_xx = signal.welch(
            np.column_stack(
                [data[a * averaging_step : -((averaging - a - 1) * averaging_step) or None] for a in range(averaging)],
            ),
            fs=rate,
            nperseg=data.size - (averaging - 1) * averaging_step,
            axis=0,
        )
        pn_xx = np.mean(pn_xx, axis=1)

    if progress:
        print(progress, end="", flush=True)

    return freq, pn_xx


def measure_noise_welch_iter(
    channel: str,
    length: int,
    *,
    count: int = 1,
    rate: float | None = None,
    reset_adc: bool = _RESET_ADC_DEFAULT,
) -> Iterator[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    if reset_adc:
        device_adc.reset_device()
    task_adc: Task
    with Task() as task_adc:
        task_adc.ai_channels.add_ai_voltage_chan(channel)
        if rate is None:
            rate = task_adc.timing.samp_clk_max_rate
        task_adc.timing.cfg_samp_clk_timing(
            rate=rate,
            sample_mode=AcquisitionType.FINITE,  # don't use continuous
            samps_per_chan=length,
        )
        for _ in range(count):
            data: NDArray[np.float64] = np.asarray(task_adc.read(length))

            freq: NDArray[np.float64]
            pn_xx: NDArray[np.float64]
            freq, pn_xx = signal.welch(data, fs=rate, nperseg=length)
            yield freq, pn_xx


def measure_noise_welch_iter_multiple(
    channel: str,
    length: int,
    *,
    count: int = 1,
    rate: float | None = None,
) -> Iterator[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    task_adc: Task
    with Task() as task_adc:
        task_adc.ai_channels.add_ai_voltage_chan(channel)
        if rate is None:
            rate = task_adc.timing.samp_clk_max_rate
        task_adc.timing.cfg_samp_clk_timing(
            rate=rate,
            sample_mode=AcquisitionType.FINITE,  # don't use continuous
            samps_per_chan=length,
        )
        for _ in range(count):
            _data: NDArray[np.float64] = np.asarray(task_adc.read(length))

            _freq: NDArray[np.float64]
            _pn_xx: NDArray[np.float64]
            _freq, _pn_xx = signal.welch(_data, fs=rate, nperseg=length)
            yield _freq, _pn_xx


def measure_noise_trend(
    channel: PhysicalChannel | Iterable[PhysicalChannel],
    duration: float,
    *,
    rate: float | None = None,
    reset_adc: bool = _RESET_ADC_DEFAULT,
    out_channel: PhysicalChannel | None = None,
    out_value: float = np.nan,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if reset_adc:
        device_adc.reset_device()
    task_adc: Task
    task_dac: Task
    c: AIChannel
    with Task() as task_adc, Task() as task_dac:
        if isinstance(channel, PhysicalChannel):
            c = task_adc.ai_channels.add_ai_voltage_chan(channel.name)
            c.ai_enhanced_alias_rejection_enable = rate is not None and rate < 1000.0
        else:
            for _channel in list(channel):
                c = task_adc.ai_channels.add_ai_voltage_chan(_channel.name)
                c.ai_enhanced_alias_rejection_enable = rate is not None and rate < 1000.0
        if out_channel is not None:
            task_dac.ao_channels.add_ao_voltage_chan(out_channel.name)
            task_dac.write(out_value)
        if rate is None:
            rate = task_adc.timing.samp_clk_max_rate
        length: int = round(duration * rate)
        task_adc.timing.cfg_samp_clk_timing(rate=rate, sample_mode=AcquisitionType.CONTINUOUS)
        data: NDArray[np.float64] = np.asarray(task_adc.read(length, timeout=WAIT_INFINITELY))

    return np.arange(0, data.size) / rate, data
