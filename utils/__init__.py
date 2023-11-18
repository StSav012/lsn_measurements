# -*- coding: utf-8 -*-
from __future__ import annotations

import sys

from .config import (
    get_bool,
    get_decimal,
    get_decimal_list,
    get_float,
    get_float_list,
    get_float_tuple,
    get_int,
    get_str,
)
from .connected_points import linear_segments, sine_segments
from .count import Count
from .decimalslice import DecimalSlice
from .files_io import load_txt, read_340_table, save_txt
from .filewriter import FileWriter
from .floatslice import FloatSlice
from .ni import (
    measure_noise_fft,
    measure_noise_trend,
    measure_noise_welch,
    measure_noise_welch_iter,
    measure_offsets,
    zero_sources,
)
from .printqueue import PrintQueue
from .processing import moving_mean, moving_median, welch
from .si import parse_temperature
from .slice_sequence import SliceSequence
from .string_utils import decimals, format_float, multi_split, nth_occurrence, seconds_to_time
from .teepath import TeePath
from .tick_strings import superscript_number, tick_strings

__all__ = [
    "Count",
    "FileWriter",
    "PrintQueue",
    "FloatSlice",
    "DecimalSlice",
    "SliceSequence",
    "Auto",
    "TeePath",
    "decimals",
    "format_float",
    "multi_split",
    "nth_occurrence",
    "parse_temperature",
    "zero_sources",
    "measure_offsets",
    "measure_noise_fft",
    "measure_noise_trend",
    "measure_noise_welch",
    "measure_noise_welch_iter",
    "seconds_to_time",
    "linear_segments",
    "sine_segments",
    "warning",
    "error",
    "get_str",
    "get_bool",
    "get_int",
    "get_float",
    "get_decimal",
    "get_float_tuple",
    "get_float_list",
    "get_decimal_list",
    "superscript_number",
    "tick_strings",
    "load_txt",
    "save_txt",
    "read_340_table",
    "welch",
    "moving_mean",
    "moving_median",
]

Auto = None


def warning(msg: str) -> None:
    sys.stderr.write(f"WARNING: {msg}\n")


def error(msg: str) -> None:
    sys.stderr.write(f"ERROR: {msg}\n")
