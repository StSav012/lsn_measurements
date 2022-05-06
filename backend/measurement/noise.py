# coding: utf-8
import time
from multiprocessing import Process, Queue
from typing import Final, Tuple

import numpy as np
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.task import Task

from backend.hardware import *
from backend.utils import *


class NoiseMeasurement(Process):
    def __init__(self, results_queue: Queue, sample_rate: float, current: float,
                 ballast_resistance: float, voltage_gain: float, current_divider: float,
                 resistance_in_series: float = 0.0) -> None:
        super(NoiseMeasurement, self).__init__()
        self.results_queue: Queue[Tuple[float, np.ndarray]] = results_queue

        self.sample_rate: Final[float] = sample_rate
        self.current: Final[float] = current
        self.ballast_resistance: Final[float] = ballast_resistance
        self.resistance_in_series: float = resistance_in_series
        self.voltage_gain: Final[float] = voltage_gain
        self.current_divider: Final[float] = current_divider

    def run(self) -> None:
        measure_offsets()
        task_adc: Task
        task_dac: Task

        with Task() as task_adc, Task() as task_dac:
            task_adc.ai_channels.add_ai_voltage_chan(adc_current.name)
            task_adc.ai_channels.add_ai_voltage_chan(adc_voltage.name)
            task_dac.ao_channels.add_ao_voltage_chan(dac_current.name)
            task_dac.write(self.current * self.current_divider * (DIVIDER_RESISTANCE + self.ballast_resistance))
            time.sleep(0.01)

            task_adc.timing.cfg_samp_clk_timing(rate=self.sample_rate,
                                                sample_mode=AcquisitionType.CONTINUOUS,
                                                samps_per_chan=round(0.1 * self.sample_rate),
                                                )

            adc_stream: AnalogMultiChannelReader = AnalogMultiChannelReader(task_adc.in_stream)

            task_adc.start()

            while adc_stream.verify_array_shape:
                data: np.ndarray = np.empty((2, task_adc.timing.samp_quant_samp_per_chan))
                adc_stream.read_many_sample(data, task_adc.timing.samp_quant_samp_per_chan)
                data[1] = (data[1] - offsets[adc_voltage.name]) / self.voltage_gain
                data[0] = (data[0] - offsets[adc_current.name] - data[1]) / self.ballast_resistance
                data[1] -= data[0] * self.resistance_in_series
                self.results_queue.put((task_adc.timing.samp_clk_rate, data))

        zero_sources()