# -*- coding: utf-8 -*-

from typing import Dict, Final

from nidaqmx.system.device import Device
from nidaqmx.system.physical_channel import PhysicalChannel
from nidaqmx.system.system import System

__all__ = [
    'R', 'R_SERIES', 'VOLTAGE_GAIN', 'DIVIDER', 'DIVIDER_RESISTANCE',
    'system',
    'device_dac', 'device_adc',
    'adc_sync', 'adc_voltage', 'adc_current',
    'dac_sync', 'dac_current', 'dac_aux',
    'offsets'
]

R: float = 200.0e3
R_SERIES: float = 0.0e3
VOLTAGE_GAIN: float = 1000.0
DIVIDER: float = 1.058
DIVIDER_RESISTANCE: float = 5.65e3

system: Final[System] = System.local()

device_adc: Final[Device] = system.devices[0]
device_dac: Final[Device] = system.devices[1]

adc_voltage: Final[PhysicalChannel] = device_adc.ai_physical_chans[1]
adc_current: Final[PhysicalChannel] = device_adc.ai_physical_chans[0]
adc_sync: Final[PhysicalChannel] = device_adc.ai_physical_chans[4]
dac_current: Final[PhysicalChannel] = device_dac.ao_physical_chans[1]
dac_aux: Final[PhysicalChannel] = device_dac.ao_physical_chans[2]
dac_sync: Final[PhysicalChannel] = device_dac.ao_physical_chans[4]

offsets: Dict[str, float] = {adc_current.name: 0.0, adc_voltage.name: 0.0}
