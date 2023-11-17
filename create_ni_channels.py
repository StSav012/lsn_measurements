# coding=utf-8
"""
Create the nidaqmx/channels.py file if there is none present
to import the channel types gracefully.
Without the file, there are warnings about accessing `_task_modules`,
which is a private module.
We still do access it, but now we do it from within the `nidaqmx` package,
so no warnings are raised.
"""

try:
    from nidaqmx import channels
except ImportError:
    from pathlib import Path

    import nidaqmx

    channels: Path = Path(nidaqmx.__file__).parent / "channels.py"
    assert not channels.exists()
    channels.write_text(
        """
__author__ = 'National Instruments'
__all__ = ['Channel', 'AIChannel', 'AOChannel', 'CIChannel', 'COChannel', 'DIChannel', 'DOChannel']


from nidaqmx._task_modules.channels.channel import Channel
from nidaqmx._task_modules.channels.ai_channel import AIChannel
from nidaqmx._task_modules.channels.ao_channel import AOChannel
from nidaqmx._task_modules.channels.ci_channel import CIChannel
from nidaqmx._task_modules.channels.co_channel import COChannel
from nidaqmx._task_modules.channels.di_channel import DIChannel
from nidaqmx._task_modules.channels.do_channel import DOChannel

"""
    )

    from nidaqmx import channels


__all__ = ["channels"]
