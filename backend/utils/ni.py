# -*- coding: utf-8 -*-
from typing import Final, Iterable, Iterator, List, Optional, Tuple, Union, cast

import numpy as np
from nidaqmx.channels import AIChannel
from nidaqmx.constants import *
from nidaqmx.system.physical_channel import PhysicalChannel
from nidaqmx.task import Task
from numpy.typing import NDArray
from scipy import signal

from backend.hardware import *

__all__ = [
    'zero_sources',
    'max_sample_rate',
    'measure_offsets',
    'measure_noise_fft', 'measure_noise_trend', 'measure_noise_welch', 'measure_noise_welch_iter',
]

_RESET_ADC_DEFAULT: Final[bool] = device_adc.name != device_dac.name

# disable the function to speed up the ADC data processing
# NDArray[np.float64].tolist = lambda a: a


def zero_sources() -> None:
    task_dac_current: Task
    with Task() as task_dac_current:
        task_dac_current.ao_channels.add_ao_voltage_chan(dac_current.name)
        task_dac_current.write(0.0)
        task_dac_current.wait_until_done()


def max_sample_rate(reset_adc: bool = True) -> float:
    if reset_adc:
        device_adc.reset_device()
    task_adc: Task
    with Task() as task_adc:
        task_adc.ai_channels.add_ai_voltage_chan(adc_current.name)
        task_adc.ai_channels.add_ai_voltage_chan(adc_voltage.name)
        return task_adc.timing.samp_clk_max_rate


def measure_offsets(duration: float = 0.04, do_zero_sources: bool = True, reset_adc: bool = True) -> None:
    if reset_adc:
        device_adc.reset_device()
    task_adc: Task
    with Task() as task_adc:
        task_adc.ai_channels.add_ai_voltage_chan(adc_current.name)
        task_adc.ai_channels.add_ai_voltage_chan(adc_voltage.name)
        count: int = round(duration * task_adc.timing.samp_clk_max_rate)
        task_adc.timing.cfg_samp_clk_timing(rate=task_adc.timing.samp_clk_max_rate,
                                            sample_mode=AcquisitionType.CONTINUOUS)
        if do_zero_sources:
            zero_sources()
        task_adc.start()
        data: List[float] = task_adc.read(count, timeout=WAIT_INFINITELY)
        task_adc.stop()
        offsets[adc_current.name] = cast(float, np.mean(data[0]))
        offsets[adc_voltage.name] = cast(float, np.mean(data[1]))


def measure_noise_fft(length: int, rate: Optional[float] = None, reset_adc: bool = _RESET_ADC_DEFAULT) \
        -> Tuple[NDArray[np.float64], Tuple[NDArray[np.float64], str], Tuple[NDArray[np.float64], str]]:
    if reset_adc:
        device_adc.reset_device()
    task_adc: Task
    with Task() as task_adc:
        # task_adc.ai_channels.add_ai_voltage_chan(adc_current.name)
        task_adc.ai_channels.add_ai_voltage_chan(adc_voltage.name)
        if rate is None:
            rate = task_adc.timing.samp_clk_max_rate
        task_adc.timing.cfg_samp_clk_timing(rate=rate,
                                            sample_mode=AcquisitionType.FINITE,  # don't use continuous
                                            samps_per_chan=length)
        zero_sources()
        data = np.array(task_adc.read(length))

        task_adc.close()

        freq: NDArray[np.float64] = np.fft.fftfreq(data.shape[1], 1.0 / rate)

        return (
            np.abs(freq),
            (np.square(np.abs(np.fft.fft(data[0], norm='ortho'))), adc_current.name),
            (np.square(np.abs(np.fft.fft(data[1], norm='ortho'))), adc_voltage.name),
        )


def measure_noise_welch(channel: str, resolution: float, rate: Optional[float] = None, *,
                        averaging: int = 1, averaging_shift: float = 0.,
                        reset_adc: bool = _RESET_ADC_DEFAULT,
                        progress: str = '') -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    if averaging < 1:
        raise ValueError('Averaging must be a positive number')

    if reset_adc:
        device_adc.reset_device()
    task_adc: Task
    with Task() as task_adc:
        task_adc.ai_channels.add_ai_voltage_chan(channel)
        if rate is None:
            rate = task_adc.timing.samp_clk_max_rate

        averaging_step: int = round(rate * averaging_shift)
        if averaging != 1 and averaging_step < 1:
            raise ValueError(f'Averaging shift must be greater than {0.5 / rate}')

        length: int = round(rate / resolution) + (averaging - 1) * averaging_step
        task_adc.timing.cfg_samp_clk_timing(rate=rate,
                                            sample_mode=AcquisitionType.CONTINUOUS)
        task_adc.start()
        if progress:
            print(progress, end='', flush=True)
        data: NDArray[np.float64] = np.array(task_adc.read(length, timeout=WAIT_INFINITELY), dtype=np.float64)
        if progress:
            print(progress, end='', flush=True)
        task_adc.stop()

    freq: NDArray[np.float64]
    pn_xx: NDArray[np.float64]
    if averaging == 1:
        freq, pn_xx = signal.welch(data, fs=rate, nperseg=data.size)
    else:
        a: int
        freq, pn_xx = signal.welch(np.column_stack([
            data[a * averaging_step:-((averaging - a - 1) * averaging_step) or None]
            for a in range(averaging)]),
                                   fs=rate, nperseg=data.size - (averaging - 1) * averaging_step, axis=0)
        pn_xx = np.mean(pn_xx, axis=1)

    if progress:
        print(progress, end='', flush=True)

    return freq, pn_xx


def measure_noise_welch_iter(channel: str, length: int, count: int = 1, rate: Optional[float] = None,
                             reset_adc: bool = _RESET_ADC_DEFAULT
                             ) -> Iterator[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    if reset_adc:
        device_adc.reset_device()
    task_adc: Task
    with Task() as task_adc:
        task_adc.ai_channels.add_ai_voltage_chan(channel)
        if rate is None:
            rate = task_adc.timing.samp_clk_max_rate
        task_adc.timing.cfg_samp_clk_timing(rate=rate,
                                            sample_mode=AcquisitionType.FINITE,  # don't use continuous
                                            samps_per_chan=length)
        for _ in range(count):
            data: NDArray[np.float64] = np.array(task_adc.read(length))

            freq: NDArray[np.float64]
            pn_xx: NDArray[np.float64]
            freq, pn_xx = signal.welch(data, fs=rate, nperseg=length)
            yield freq, pn_xx
        print('done')


# def measure_noise_welch_iter_multiple(channel: str, length: int, count: int = 1, rate: Optional[float] = None) \
#         -> Iterator[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
#     task_adc: Task
#     with Task() as task_adc:
#         task_adc.ai_channels.add_ai_voltage_chan(channel)
#         if rate is None:
#             rate = task_adc.timing.samp_clk_max_rate
#         task_adc.timing.cfg_samp_clk_timing(rate=rate,
#                                             sample_mode=AcquisitionType.FINITE,  # don't use continuous
#                                             samps_per_chan=length)
#         for _ in range(count):
#             _data: NDArray[np.float64] = np.array(task_adc.read(length))
#
#             _freq: NDArray[np.float64]
#             _pn_xx: NDArray[np.float64]
#             _freq, _pn_xx = signal.welch(_data, fs=rate, nperseg=length)
#             yield _freq, _pn_xx
#         print('done')


def measure_noise_trend(channel: Union[PhysicalChannel, Iterable[PhysicalChannel]], duration: float,
                        rate: Optional[float] = None,
                        reset_adc: bool = _RESET_ADC_DEFAULT,
                        out_channel: Optional[PhysicalChannel] = None, out_value: float = np.nan) \
        -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
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
            for channel in list(channel):
                c = task_adc.ai_channels.add_ai_voltage_chan(channel.name)
                c.ai_enhanced_alias_rejection_enable = rate is not None and rate < 1000.0
        if out_channel is not None:
            task_dac.ao_channels.add_ao_voltage_chan(out_channel.name)
            task_dac.write(out_value)
        if rate is None:
            rate = task_adc.timing.samp_clk_max_rate
        length: int = round(duration * rate)
        task_adc.timing.cfg_samp_clk_timing(rate=rate,
                                            sample_mode=AcquisitionType.CONTINUOUS)
        data: NDArray[np.float64] = task_adc.read(length, timeout=WAIT_INFINITELY)

    return np.arange(0, data.size) / rate, data
