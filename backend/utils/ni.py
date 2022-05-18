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
    'measure_offsets',
    'measure_noise_fft', 'measure_noise_trend', 'measure_noise_welch', 'measure_noise_welch_iter',
]

_RESET_ADC_DEFAULT: Final[bool] = device_adc.name != device_dac.name

# disable the function to speed up the ADC data processing
# NDArray[np.float64].tolist = lambda a: a


def zero_sources(reset_dac: bool = True) -> None:
    if reset_dac:
        device_dac.reset_device()
    task_dac: Task
    with Task() as task_dac:
        channel: PhysicalChannel
        for channel in device_dac.ao_physical_chans:
            task_dac.ao_channels.add_ao_voltage_chan(channel.name)
        task_dac.write([0.0] * len(device_dac.ao_physical_chans))
        task_dac.wait_until_done()


def measure_offsets(duration: float = 0.04, do_zero_sources: bool = True, reset_adc: bool = True) -> None:
    if reset_adc:
        device_adc.reset_device()
    task_adc: Task
    with Task() as task_adc:
        channel: PhysicalChannel
        for channel in device_adc.ai_physical_chans:
            task_adc.ai_channels.add_ai_voltage_chan(channel.name)
        count: int = round(duration * task_adc.timing.samp_clk_max_rate)
        task_adc.timing.cfg_samp_clk_timing(rate=task_adc.timing.samp_clk_max_rate,
                                            sample_mode=AcquisitionType.CONTINUOUS)
        if do_zero_sources:
            zero_sources()
        task_adc.start()
        data: List[float] = task_adc.read(count, timeout=WAIT_INFINITELY)
        task_adc.stop()
        index: int
        for index, channel in enumerate(device_adc.ai_physical_chans):
            offsets[channel.name] = cast(float, np.mean(data[index]))


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


if not hasattr(Task, 'output_onboard_buffer_size'):
    import ctypes
    import nidaqmx
    from _ctypes import CFuncPtr
    from nidaqmx.errors import check_for_error

    def get_output_onboard_buffer_size(self) -> int:
        """
        int: Indicates in samples per channel the size of the onboard output buffer of the device.
        """
        val: ctypes.c_uint32 = ctypes.c_uint32()

        lib_importer = getattr(nidaqmx, '_lib').lib_importer
        c_func: CFuncPtr = lib_importer.windll.DAQmxGetBufOutputOnbrdBufSize
        if c_func.argtypes is None:
            with c_func.arglock:
                if c_func.argtypes is None:
                    c_func.argtypes = [lib_importer.task_handle, ctypes.POINTER(ctypes.c_uint)]

        error_code: int = c_func(self._handle, ctypes.byref(val))
        check_for_error(error_code)

        return val.value

    def set_output_onboard_buffer_size(self, buffer_size: int) -> None:
        """
        int: Specifies in samples per channel the size of the onboard output buffer of the device.
        """
        val: ctypes.c_uint32 = ctypes.c_uint32(buffer_size)

        lib_importer = getattr(nidaqmx, '_lib').lib_importer
        c_func: CFuncPtr = lib_importer.windll.DAQmxSetBufOutputOnbrdBufSize
        if c_func.argtypes is None:
            with c_func.arglock:
                if c_func.argtypes is None:
                    c_func.argtypes = [lib_importer.task_handle, ctypes.POINTER(ctypes.c_uint32)]

        error_code: int = c_func(self._handle, val)
        check_for_error(error_code)

    Task.output_onboard_buffer_size = property(
        fget=get_output_onboard_buffer_size,
        fset=set_output_onboard_buffer_size,
        fdel=lambda self: None,
        doc='int: Specifies in samples per channel the size of the onboard output buffer of the device.')


if not hasattr(Task, 'input_onboard_buffer_size'):
    import ctypes
    import nidaqmx
    from _ctypes import CFuncPtr
    from nidaqmx.errors import check_for_error

    def get_input_onboard_buffer_size(self) -> int:
        """
        int: Indicates in samples per channel the size of the onboard input buffer of the device.
        """
        val: ctypes.c_uint = ctypes.c_uint()

        lib_importer = getattr(nidaqmx, '_lib').lib_importer
        c_func: CFuncPtr = lib_importer.windll.DAQmxGetBufOutputOnbrdBufSize
        if c_func.argtypes is None:
            with c_func.arglock:
                if c_func.argtypes is None:
                    c_func.argtypes = [lib_importer.task_handle, ctypes.POINTER(ctypes.c_uint32)]

        error_code: int = c_func(self._handle, ctypes.byref(val))
        check_for_error(error_code)

        return val.value

    Task.input_onboard_buffer_size = property(
        fget=get_input_onboard_buffer_size,
        fdel=lambda self: None,
        doc='int: Indicates in samples per channel the size of the onboard input buffer of the device.')
