# -*- coding: utf-8 -*-
from __future__ import annotations

import functools
from typing import Sequence

__all__ = ['decimals', 'format_float', 'seconds_to_time', 'multi_split', 'nth_occurrence']


def decimals(text: str) -> int:
    if '.' in text:
        return len(text) - text.rfind('.') - 1
    else:
        return 0


@functools.lru_cache(maxsize=128, typed=True)
def format_float(value: float, precision: int = 12, *, prefix: str = '', suffix: str = '') -> str:
    return prefix + f'{value:.{precision}f}'.rstrip('0').rstrip('.') + suffix


def seconds_to_time(seconds: int) -> str:
    if not seconds:
        return '00:00:00'
    hours: int = abs(seconds) // 3600
    minutes: int = (abs(seconds) - hours * 3600) // 60
    seconds: int = abs(seconds) % 60
    return f'{"-" if seconds < 0 else ""}{hours:02d}:{minutes:02d}:{seconds:02d}'


def multi_split(text: str, separators: Sequence[str]) -> list[str]:
    words: list[str] = []
    separator: str
    index: int = -1
    while index < len(text):
        index += 1
        for separator in separators:
            if text.startswith(separator, index):
                words.append(text[:index])
                text = text[index+len(separator):]
                index = 0
    words.append(text)
    return words


def nth_occurrence(text: str, substrings: str | Sequence[str], n: int) -> int:
    if isinstance(substrings, str):
        substrings = (substrings, )
    index: int
    substring: str
    count: int = 0
    for index in range(len(text)):
        if any(text.startswith(substring, index) for substring in substrings):
            count += 1
            if count == n:
                return index
    return -1
