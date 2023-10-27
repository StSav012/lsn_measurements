# -*- coding: utf-8 -*-
from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np

__all__ = [
    "SI_PREFIX_EXPONENTS",
    "parse_temperature",
    "si_prefix",
    "si_factor",
    "parse_si_number",
]

SI_PREFIX_EXPONENTS: dict[str, int] = {
    "y": -24,
    "z": -21,
    "a": -18,
    "f": -15,
    "p": -12,
    "n": -9,
    "µ": -6,
    "u": -6,
    "m": -3,
    "c": -2,
    "d": -1,
    "": 0,
    "da": 1,
    "h": 2,
    "k": 3,
    "M": 6,
    "G": 9,
    "T": 12,
    "P": 15,
    "E": 18,
    "Z": 21,
    "Y": 24,
}


def parse_temperature(text: str) -> float:
    text = text.strip()
    if not text.endswith("K"):
        return np.nan
    text = text[:-1]
    for k, v in SI_PREFIX_EXPONENTS.items():
        if k and text.endswith(k):
            return float(text[: -len(k)].strip()) * (10.0**v)
    try:
        return float(text)
    except ValueError:
        return np.nan


@functools.lru_cache(maxsize=128, typed=True)
def si_prefix(unit: str) -> str:
    matching_prefixes: list[str] = [prefix for prefix in SI_PREFIX_EXPONENTS if unit.startswith(prefix)]
    if not matching_prefixes:
        return ""
    else:
        return sorted(matching_prefixes, key=len, reverse=True)[0]


@functools.lru_cache(maxsize=128, typed=True)
def si_factor(unit: str) -> float:
    return 10 ** SI_PREFIX_EXPONENTS[si_prefix(unit)]


@functools.lru_cache(maxsize=128, typed=True)
def parse_si_number(number_text: Any) -> float:
    if isinstance(number_text, float):
        return number_text
    if not isinstance(number_text, str):
        return math.nan
    leftovers: int = len(number_text)
    while leftovers > 0:
        try:
            float(number_text[:leftovers])
        except ValueError:
            leftovers -= 1
        else:
            return float(number_text[:leftovers]) * si_factor(number_text[leftovers:].strip())
    else:
        return math.nan
