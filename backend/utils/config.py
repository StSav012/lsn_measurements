# -*- coding: utf-8 -*-
from __future__ import annotations

from configparser import ConfigParser
from decimal import Decimal
from typing import Any, Sequence

from backend.utils.si import parse_si_number

__all__ = [
    "get_str",
    "get_bool",
    "get_int",
    "get_float",
    "get_decimal",
    "get_float_tuple",
    "get_float_list",
    "get_decimal_list",
]

__sentinel = object()


def get(
    config: ConfigParser,
    sample: str,
    section: str,
    key: str,
    fallback: Any = __sentinel,
) -> str:
    if f"{section}/{sample}" in config.sections():
        if key in config[f"{section}/{sample}"]:
            if fallback is not __sentinel:
                return config.get(f"{section}/{sample}", key, fallback=fallback)
            else:
                return config.get(f"{section}/{sample}", key)
    if fallback is not __sentinel:
        return config.get(f"{section}", key, fallback=fallback)
    else:
        return config.get(f"{section}", key)


def get_str(
    config: ConfigParser,
    sample: str,
    section: str,
    key: str,
    fallback: str = __sentinel,
) -> str:
    return get(config=config, sample=sample, section=section, key=key, fallback=fallback)


def get_bool(
    config: ConfigParser,
    sample: str,
    section: str,
    key: str,
    fallback: bool = __sentinel,
) -> bool:
    if f"{section}/{sample}" in config.sections():
        if key in config[f"{section}/{sample}"]:
            if fallback is not __sentinel:
                return config.getboolean(f"{section}/{sample}", key, fallback=fallback)
            else:
                return config.getboolean(f"{section}/{sample}", key)
    if fallback is not __sentinel:
        return config.getboolean(f"{section}", key, fallback=fallback)
    else:
        return config.getboolean(f"{section}", key)


def get_int(
    config: ConfigParser,
    sample: str,
    section: str,
    key: str,
    fallback: int = __sentinel,
) -> int:
    if f"{section}/{sample}" in config.sections():
        if key in config[f"{section}/{sample}"]:
            if fallback is not __sentinel:
                return config.getint(f"{section}/{sample}", key, fallback=fallback)
            else:
                return config.getint(f"{section}/{sample}", key)
    if fallback is not __sentinel:
        return config.getint(f"{section}", key, fallback=fallback)
    else:
        return config.getint(f"{section}", key)


def get_float(
    config: ConfigParser,
    sample: str,
    section: str,
    key: str,
    fallback: float = __sentinel,
) -> float:
    if f"{section}/{sample}" in config.sections():
        if key in config[f"{section}/{sample}"]:
            if fallback is not __sentinel:
                return parse_si_number(config.get(f"{section}/{sample}", key, fallback=fallback))
            else:
                return parse_si_number(config.get(f"{section}/{sample}", key))
    if fallback is not __sentinel:
        return parse_si_number(config.get(f"{section}", key, fallback=fallback))
    else:
        return parse_si_number(config.get(f"{section}", key))


def get_decimal(
    config: ConfigParser,
    sample: str,
    section: str,
    key: str,
    fallback: float = __sentinel,
) -> Decimal:
    return Decimal.from_float(get_float(config=config, sample=sample, section=section, key=key, fallback=fallback))


def get_float_tuple(
    config: ConfigParser,
    sample: str,
    section: str,
    key: str,
    fallback: Sequence[float] = __sentinel,
    separator: str = ",",
) -> tuple[float, ...]:
    return tuple(
        map(
            float,
            get(
                config=config,
                sample=sample,
                section=section,
                key=key,
                fallback=fallback,
            ).split(separator),
        )
    )


def get_float_list(
    config: ConfigParser,
    sample: str,
    section: str,
    key: str,
    fallback: Sequence[float] = __sentinel,
    separator: str = ",",
) -> list[float]:
    return list(
        map(
            float,
            get(
                config=config,
                sample=sample,
                section=section,
                key=key,
                fallback=fallback,
            ).split(separator),
        )
    )


def get_decimal_list(
    config: ConfigParser,
    sample: str,
    section: str,
    key: str,
    fallback: Sequence[float] = __sentinel,
    separator: str = ",",
) -> list[Decimal]:
    return list(
        map(
            Decimal,
            get(
                config=config,
                sample=sample,
                section=section,
                key=key,
                fallback=fallback,
            ).split(separator),
        )
    )
