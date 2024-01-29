# -*- coding: utf-8 -*-
from __future__ import annotations

from configparser import ConfigParser
from decimal import Decimal
from typing import Any, LiteralString, Sequence

from .si import parse_si_number
from .slice_sequence import SliceSequence

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
    section: LiteralString,
    key: LiteralString,
    *,
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
    section: LiteralString,
    key: LiteralString,
    *,
    fallback: str = __sentinel,
) -> str:
    return get(config=config, sample=sample, section=section, key=key, fallback=fallback)


def get_bool(
    config: ConfigParser,
    sample: str,
    section: LiteralString,
    key: LiteralString,
    *,
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
    section: LiteralString,
    key: LiteralString,
    *,
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
    section: LiteralString,
    key: LiteralString,
    *,
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
    section: LiteralString,
    key: LiteralString,
    *,
    fallback: Decimal | float | str | tuple[int, Sequence[int], int] = __sentinel,
) -> Decimal:
    return Decimal(get(config=config, sample=sample, section=section, key=key, fallback=fallback))


def get_float_tuple(
    config: ConfigParser,
    sample: str,
    section: LiteralString,
    key: LiteralString,
    *,
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
    section: LiteralString,
    key: LiteralString,
    *,
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
    section: LiteralString,
    key: LiteralString,
    *,
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


def get_slice_sequence(
    config: ConfigParser,
    sample: str,
    section: LiteralString,
    key: LiteralString,
    *,
    fallback: SliceSequence = __sentinel,
    slice_separator: str | Sequence[str] = ("..", ":"),
    items_separator: str | Sequence[str] = (",", ";"),
) -> SliceSequence:
    if f"{section}/{sample}" in config.sections():
        if key in config[f"{section}/{sample}"]:
            if fallback is not __sentinel:
                return SliceSequence(
                    config.get(f"{section}/{sample}", key, fallback=fallback),
                    slice_separator=slice_separator,
                    items_separator=items_separator,
                )
            else:
                return SliceSequence(
                    config.get(f"{section}/{sample}", key),
                    slice_separator=slice_separator,
                    items_separator=items_separator,
                )
    if fallback is not __sentinel:
        return SliceSequence(
            config.get(f"{section}", key, fallback=fallback),
            slice_separator=slice_separator,
            items_separator=items_separator,
        )
    else:
        return SliceSequence(
            config.get(f"{section}", key),
            slice_separator=slice_separator,
            items_separator=items_separator,
        )
