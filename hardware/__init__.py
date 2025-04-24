try:
    from tomllib import loads
except ImportError:
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from tomli import loads

from inspect import getfullargspec
from pathlib import Path
from typing import Any, Final

from nidaqmx.constants import ChannelType, FillMode, UsageTypeAI
from nidaqmx.system.device import Device
from nidaqmx.system.physical_channel import PhysicalChannel
from nidaqmx.system.system import System
from nidaqmx.task import Task
from numpy import float64, zeros
from numpy.typing import NDArray

__all__ = [
    "R",
    "R_SERIES",
    "VOLTAGE_GAIN",
    "DIVIDER",
    "DIVIDER_RESISTANCE",
    "device_dac",
    "device_adc",
    "device_dio",
    "adc_sync",
    "adc_voltage",
    "adc_current",
    "dac_sync",
    "dac_current",
    "dac_aux",
    "dac_synth_pulse",
    "offsets",
]

config_file: Path = Path(__file__).parent / "config.toml"
if not config_file.exists():
    from utils.tkinter_message import show_error

    show_error(
        title="No hardware config found",
        message="""There should be a file named “config.toml” with the hardware description:

[PXI]
ADC = "PXI-..."
DAC = "PXI-..."
DIO = "PXI-..."

[physical_chans.AI]
voltage = <int>
current = <int>
sync = <int>

[physical_chans.AO]
current = <int>
aux = <int>
synth_pulse = <int>
sync = <int>

[circuitry]
R = <float>
R_SERIES = <float>
VOLTAGE_GAIN = <float>
DIVIDER = <float>
DIVIDER_RESISTANCE = <float>
""",
    )
    exit(1)

config: dict[str, Any] = loads(config_file.read_text(encoding="utf-8"))

pxi: Final[dict[str, float]] = config["PXI"]

circuitry: Final[dict[str, float]] = config["circuitry"]
R: float = circuitry["R"]
R_SERIES: float = circuitry["R_SERIES"]
VOLTAGE_GAIN: float = circuitry["VOLTAGE_GAIN"]
DIVIDER: float = circuitry["DIVIDER"]
DIVIDER_RESISTANCE: float = circuitry["DIVIDER_RESISTANCE"]

system: Final[System] = System.local()


def find_device(**kwargs: Any) -> Device:
    device: Device
    for device in system.devices:
        if all(getattr(device, key) == value for key, value in kwargs.items()):
            device.reset_device()
            return device
    raise LookupError(f"No device matching {kwargs} found")


device_adc: Final[Device] = find_device(product_type=pxi["ADC"])
device_dac: Final[Device] = find_device(product_type=pxi["DAC"])
device_dio: Final[Device] = find_device(product_type=pxi["DIO"])

physical_chans: Final[dict[str, dict[str, int]]] = config["physical_chans"]

adc_voltage: Final[PhysicalChannel] = device_adc.ai_physical_chans[physical_chans["AI"]["voltage"]]
adc_current: Final[PhysicalChannel] = device_adc.ai_physical_chans[physical_chans["AI"]["current"]]
adc_sync: Final[PhysicalChannel] = device_adc.ai_physical_chans[physical_chans["AI"]["sync"]]
dac_current: Final[PhysicalChannel] = device_dac.ao_physical_chans[physical_chans["AO"]["current"]]
dac_aux: Final[PhysicalChannel] = device_dac.ao_physical_chans[physical_chans["AO"]["aux"]]
dac_synth_pulse: Final[PhysicalChannel] = device_dac.ao_physical_chans[physical_chans["AO"]["synth_pulse"]]
dac_sync: Final[PhysicalChannel] = device_dac.ao_physical_chans[physical_chans["AO"]["sync"]]

offsets: dict[str, float] = {adc_current.name: 0.0, adc_voltage.name: 0.0}


# don't convert the samples read from NDArray into a list
# the following function is an almost exact copy of what is in the NI sources, except there is no `np.tolist` used
NUM_SAMPLES_UNSET, default_timeout = getfullargspec(Task.read).defaults


def _read_ai_faster(
    self: Task,
    number_of_samples_per_channel=NUM_SAMPLES_UNSET,
    timeout: float = default_timeout,
) -> float64 | NDArray[float64]:
    channels_to_read = self.in_stream.channels_to_read
    read_chan_type: ChannelType = channels_to_read.chan_type

    if read_chan_type != ChannelType.ANALOG_INPUT or any(
        chan.ai_meas_type == UsageTypeAI.POWER for chan in channels_to_read
    ):
        # use the NI function, backed up as `_read` prior to `_read_ai_faster` function use
        # noinspection PyUnresolvedReferences
        return self._read(number_of_samples_per_channel, timeout)

    num_samples_not_set: bool = number_of_samples_per_channel is NUM_SAMPLES_UNSET
    number_of_samples_per_channel: int = self._calculate_num_samps_per_chan(number_of_samples_per_channel)
    number_of_channels: int = len(channels_to_read.channel_names)

    # Determine the array shape and size to create
    if number_of_channels > 1:
        if not num_samples_not_set:
            array_shape = (number_of_channels, number_of_samples_per_channel)
        else:
            array_shape = number_of_channels
    else:
        array_shape = number_of_samples_per_channel

    # Analog Input Only
    data: NDArray[float64] = zeros(array_shape, dtype=float64)
    samples_read: int
    _, samples_read = self._interpreter.read_analog_f64(
        self._handle, number_of_samples_per_channel, timeout, FillMode.GROUP_BY_CHANNEL.value, data
    )

    if num_samples_not_set and array_shape == 1:
        return data[0]

    if samples_read != number_of_samples_per_channel:
        if number_of_channels > 1:
            return data[:, :samples_read]
        else:
            return data[:samples_read]

    return data


Task._read = Task.read
Task.read = _read_ai_faster
