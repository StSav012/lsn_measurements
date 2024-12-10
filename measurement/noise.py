# coding: utf-8
import time
from multiprocessing import Event, Process, Queue
from typing import Final

import numpy as np
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.system.physical_channel import PhysicalChannel
from nidaqmx.task import Task
from numpy.typing import NDArray

from hardware import (
    DIVIDER_RESISTANCE,
    adc_current,
    adc_voltage,
    dac_current,
    offsets,
)
from utils.ni import measure_offsets

__all__ = ["NoiseMeasurement", "IVNoiseMeasurement"]


class NoiseMeasurement(Process):
    def __init__(
        self,
        results_queue: Queue,
        channel: PhysicalChannel,
        sample_rate: float,
        scale: float,
        measure_offset: bool = False,
    ) -> None:
        super().__init__()
        self.results_queue: Queue[tuple[float, NDArray[np.float64]]] = results_queue

        self.channel: PhysicalChannel = channel
        self.sample_rate: Final[float] = sample_rate
        self.scale: Final[float] = scale
        self.measure_offset: Final[bool] = measure_offset

        self._done: Event = Event()

    def stop(self) -> None:
        self._done.set()

    def run(self) -> None:
        self._done.clear()

        if self.measure_offset:
            measure_offsets()

        task_adc: Task

        with Task() as task_adc:
            task_adc.ai_channels.add_ai_voltage_chan(self.channel.name)

            task_adc.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=task_adc.in_stream.input_onbrd_buf_size,
            )

            adc_stream: AnalogMultiChannelReader = AnalogMultiChannelReader(task_adc.in_stream)

            number_of_channels: int = task_adc.number_of_channels
            number_of_samples_per_channel: int = task_adc.timing.samp_quant_samp_per_chan
            sample_clock_rate: float = task_adc.timing.samp_clk_rate

            task_adc.start()

            while not self._done.is_set():
                data: NDArray[np.float64] = np.empty((number_of_channels, number_of_samples_per_channel))
                adc_stream.read_many_sample(data, number_of_samples_per_channel)
                self.results_queue.put((sample_clock_rate, data))


class IVNoiseMeasurement(Process):
    def __init__(
        self,
        results_queue: Queue,
        sample_rate: float,
        current: float,
        ballast_resistance: float,
        voltage_gain: float,
        current_divider: float,
        resistance_in_series: float = 0.0,
    ) -> None:
        super(IVNoiseMeasurement, self).__init__()
        self.results_queue: Queue[tuple[float, NDArray[np.float64]]] = results_queue

        self.sample_rate: Final[float] = sample_rate
        self.current: Final[float] = current
        self.ballast_resistance: Final[float] = ballast_resistance
        self.resistance_in_series: float = resistance_in_series
        self.voltage_gain: Final[float] = voltage_gain
        self.current_divider: Final[float] = current_divider

        self._done: Event = Event()

    def stop(self) -> None:
        self._done.set()

    def run(self) -> None:
        self._done.clear()

        measure_offsets()
        task_adc: Task
        task_dac: Task

        with Task() as task_adc, Task() as task_dac:
            task_adc.ai_channels.add_ai_voltage_chan(adc_current.name)
            task_adc.ai_channels.add_ai_voltage_chan(adc_voltage.name)
            task_dac.ao_channels.add_ao_voltage_chan(dac_current.name)
            task_dac.write(self.current * self.current_divider * (DIVIDER_RESISTANCE + self.ballast_resistance))
            time.sleep(0.01)

            task_adc.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=task_adc.in_stream.input_onbrd_buf_size,
            )

            adc_stream: AnalogMultiChannelReader = AnalogMultiChannelReader(task_adc.in_stream)

            number_of_channels: int = task_adc.number_of_channels
            number_of_samples_per_channel: int = task_adc.timing.samp_quant_samp_per_chan
            sample_clock_rate: float = task_adc.timing.samp_clk_rate

            task_adc.start()

            while not self._done.is_set():
                data: NDArray[np.float64] = np.empty((number_of_channels, number_of_samples_per_channel))
                adc_stream.read_many_sample(data, number_of_samples_per_channel)
                data[1] = (data[1] - offsets[adc_voltage.name]) / self.voltage_gain
                data[0] = (data[0] - offsets[adc_current.name] - data[1]) / self.ballast_resistance
                data[1] -= data[0] * self.resistance_in_series
                self.results_queue.put((sample_clock_rate, data))
