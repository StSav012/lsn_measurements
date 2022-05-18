# -*- coding: utf-8 -*-
from configparser import ConfigParser
from decimal import Decimal
from typing import List, Optional, Sequence, Tuple

from backend.utils.si import parse_si_number

__all__ = [
    'get_str', 'get_bool', 'get_int', 'get_float', 'get_decimal',
    'get_float_tuple', 'get_float_list', 'get_decimal_list',
]


def get_str(config: ConfigParser, sample: str, section: str, key: str, fallback: Optional[str] = None) -> str:
    if f'{section}/{sample}' in config.sections():
        if key in config[f'{section}/{sample}']:
            if fallback is not None:
                return config.get(f'{section}/{sample}', key, fallback=fallback)
            else:
                return config.get(f'{section}/{sample}', key)
    if fallback is not None:
        return config.get(f'{section}', key, fallback=fallback)
    else:
        return config.get(f'{section}', key)


def get_bool(config: ConfigParser, sample: str, section: str, key: str, fallback: Optional[bool] = None) -> bool:
    if f'{section}/{sample}' in config.sections():
        if key in config[f'{section}/{sample}']:
            if fallback is not None:
                return config.getboolean(f'{section}/{sample}', key, fallback=fallback)
            else:
                return config.getboolean(f'{section}/{sample}', key)
    if fallback is not None:
        return config.getboolean(f'{section}', key, fallback=fallback)
    else:
        return config.getboolean(f'{section}', key)


def get_int(config: ConfigParser, sample: str, section: str, key: str, fallback: Optional[int] = None) -> int:
    if f'{section}/{sample}' in config.sections():
        if key in config[f'{section}/{sample}']:
            if fallback is not None:
                return config.getint(f'{section}/{sample}', key, fallback=fallback)
            else:
                return config.getint(f'{section}/{sample}', key)
    if fallback is not None:
        return config.getint(f'{section}', key, fallback=fallback)
    else:
        return config.getint(f'{section}', key)


def get_float(config: ConfigParser, sample: str, section: str, key: str, fallback: Optional[float] = None) -> float:
    if f'{section}/{sample}' in config.sections():
        if key in config[f'{section}/{sample}']:
            if fallback is not None:
                return parse_si_number(config.get(f'{section}/{sample}', key, fallback=fallback))
            else:
                return parse_si_number(config.get(f'{section}/{sample}', key))
    if fallback is not None:
        return parse_si_number(config.get(f'{section}', key, fallback=fallback))
    else:
        return parse_si_number(config.get(f'{section}', key))


def get_decimal(config: ConfigParser, sample: str, section: str, key: str, fallback: Optional[float] = None) -> Decimal:
    return Decimal.from_float(get_float(config=config, sample=sample, section=section, key=key, fallback=fallback))


def get_float_tuple(config: ConfigParser, sample: str, section: str, key: str,
                    fallback: Optional[Sequence[float]] = None,
                    separator: str = ',') -> Tuple[float]:
    return tuple(map(float, get_str(config=config, sample=sample, section=section, key=key,
                                    fallback=fallback).split(separator)))


def get_float_list(config: ConfigParser, sample: str, section: str, key: str,
                   fallback: Optional[Sequence[float]] = None,
                   separator: str = ',') -> List[float]:
    return list(map(float, get_str(config=config, sample=sample, section=section, key=key,
                                   fallback=fallback).split(separator)))


def get_decimal_list(config: ConfigParser, sample: str, section: str, key: str,
                     fallback: Optional[Sequence[float]] = None,
                     separator: str = ',') -> List[Decimal]:
    return list(map(Decimal, get_str(config=config, sample=sample, section=section, key=key,
                                     fallback=fallback).split(separator)))
