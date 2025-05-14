from collections.abc import Iterable, Sequence
from configparser import ConfigParser
from decimal import Decimal
from os import PathLike
from typing import LiteralString, overload

from .si import parse_si_number
from .slice_sequence import SliceSequence

__all__ = [
    "Config",
    "get_bool",
    "get_decimal",
    "get_decimal_list",
    "get_float",
    "get_float_list",
    "get_float_tuple",
    "get_int",
    "get_str",
]

__sentinel = object()


def get(
    config: ConfigParser,
    sample: str,
    section: LiteralString,
    key: LiteralString,
    *,
    fallback: str = __sentinel,
) -> str:
    if f"{section}/{sample}" in config.sections() and key in config[f"{section}/{sample}"]:
        if fallback is not __sentinel:
            return config.get(f"{section}/{sample}", key, fallback=fallback)
        return config.get(f"{section}/{sample}", key)
    if fallback is not __sentinel:
        return config.get(f"{section}", key, fallback=fallback)
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
    if f"{section}/{sample}" in config.sections() and key in config[f"{section}/{sample}"]:
        if fallback is not __sentinel:
            return config.getboolean(f"{section}/{sample}", key, fallback=fallback)
        return config.getboolean(f"{section}/{sample}", key)
    if fallback is not __sentinel:
        return config.getboolean(f"{section}", key, fallback=fallback)
    return config.getboolean(f"{section}", key)


def get_int(
    config: ConfigParser,
    sample: str,
    section: LiteralString,
    key: LiteralString,
    *,
    fallback: int = __sentinel,
) -> int:
    if f"{section}/{sample}" in config.sections() and key in config[f"{section}/{sample}"]:
        if fallback is not __sentinel:
            return config.getint(f"{section}/{sample}", key, fallback=fallback)
        return config.getint(f"{section}/{sample}", key)
    if fallback is not __sentinel:
        return config.getint(f"{section}", key, fallback=fallback)
    return config.getint(f"{section}", key)


def get_float(
    config: ConfigParser,
    sample: str,
    section: LiteralString,
    key: LiteralString,
    *,
    fallback: float = __sentinel,
) -> float:
    if f"{section}/{sample}" in config.sections() and key in config[f"{section}/{sample}"]:
        if fallback is not __sentinel:
            return parse_si_number(config.get(f"{section}/{sample}", key, fallback=fallback))
        return parse_si_number(config.get(f"{section}/{sample}", key))
    if fallback is not __sentinel:
        return parse_si_number(config.get(f"{section}", key, fallback=fallback))
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
    try:
        return tuple(
            map(
                float,
                get(
                    config=config,
                    sample=sample,
                    section=section,
                    key=key,
                ).split(separator),
            ),
        )
    except LookupError:
        if fallback is not __sentinel:
            return tuple(fallback)
        raise


def get_float_list(
    config: ConfigParser,
    sample: str,
    section: LiteralString,
    key: LiteralString,
    *,
    fallback: Sequence[float] = __sentinel,
    separator: str = ",",
) -> list[float]:
    try:
        return list(
            map(
                float,
                get(
                    config=config,
                    sample=sample,
                    section=section,
                    key=key,
                ).split(separator),
            ),
        )
    except LookupError:
        if fallback is not __sentinel:
            return list(fallback)
        raise


def get_decimal_list(
    config: ConfigParser,
    sample: str,
    section: LiteralString,
    key: LiteralString,
    *,
    fallback: Sequence[float] = __sentinel,
    separator: str = ",",
) -> list[Decimal]:
    try:
        return list(
            map(
                Decimal,
                get(
                    config=config,
                    sample=sample,
                    section=section,
                    key=key,
                ).split(separator),
            ),
        )
    except LookupError:
        if fallback is not __sentinel:
            return list(map(Decimal, fallback))
        raise


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
    if f"{section}/{sample}" in config.sections() and key in config[f"{section}/{sample}"]:
        if fallback is not __sentinel:
            return SliceSequence(
                config.get(f"{section}/{sample}", key, fallback=fallback),
                slice_separator=slice_separator,
                items_separator=items_separator,
            )
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
    return SliceSequence(
        config.get(f"{section}", key),
        slice_separator=slice_separator,
        items_separator=items_separator,
    )


class Config(ConfigParser):
    def __init__(
        self,
        filenames: (
            str | bytes | PathLike[str] | PathLike[bytes] | Iterable[str | bytes | PathLike[str] | PathLike[bytes]]
        ) = "config.ini",
    ) -> None:
        super().__init__(
            allow_no_value=True,
            inline_comment_prefixes=("#", ";"),
        )

        self.read(filenames=filenames)

        self._sample_name: str = self.get(section="circuitry", option="sample name")

    @property
    def sample_name(self) -> str:
        return self._sample_name

    @overload
    def get_str(self, section: LiteralString, key: LiteralString) -> str: ...

    @overload
    def get_str(self, section: LiteralString, key: LiteralString, *, fallback: str) -> str: ...

    def get_str(self, section: LiteralString, key: LiteralString, **kwargs: str) -> str:
        return get_str(self, self._sample_name, section, key, **kwargs)

    @overload
    def get_bool(self, section: LiteralString, key: LiteralString) -> bool: ...

    @overload
    def get_bool(self, section: LiteralString, key: LiteralString, *, fallback: bool) -> bool: ...

    def get_bool(self, section: LiteralString, key: LiteralString, **kwargs: bool) -> bool:
        return get_bool(self, self._sample_name, section, key, **kwargs)

    @overload
    def get_int(self, section: LiteralString, key: LiteralString) -> int: ...

    @overload
    def get_int(self, section: LiteralString, key: LiteralString, *, fallback: int) -> int: ...

    def get_int(self, section: LiteralString, key: LiteralString, **kwargs) -> int:
        return get_int(self, self._sample_name, section, key, **kwargs)

    @overload
    def get_float(self, section: LiteralString, key: LiteralString) -> float: ...

    @overload
    def get_float(self, section: LiteralString, key: LiteralString, *, fallback: float) -> float: ...

    def get_float(self, section: LiteralString, key: LiteralString, **kwargs: float) -> float:
        return get_float(self, self._sample_name, section, key, **kwargs)

    @overload
    def get_decimal(self, section: LiteralString, key: LiteralString) -> Decimal: ...

    @overload
    def get_decimal(self, section: LiteralString, key: LiteralString, *, fallback: float) -> Decimal: ...

    def get_decimal(self, section: LiteralString, key: LiteralString, **kwargs: float) -> Decimal:
        return get_decimal(self, self._sample_name, section, key, **kwargs)

    @overload
    def get_float_tuple(
        self,
        section: LiteralString,
        key: LiteralString,
        *,
        separator: str = ",",
    ) -> tuple[float, ...]: ...

    @overload
    def get_float_tuple(
        self,
        section: LiteralString,
        key: LiteralString,
        *,
        fallback: Sequence[float],
        separator: str = ",",
    ) -> tuple[float, ...]: ...

    def get_float_tuple(
        self,
        section: LiteralString,
        key: LiteralString,
        *,
        separator: str = ",",
        **kwargs: Sequence[float],
    ) -> tuple[float, ...]:
        return get_float_tuple(self, self._sample_name, section, key, separator=separator, **kwargs)

    @overload
    def get_float_list(self, section: LiteralString, key: LiteralString, *, separator: str = ",") -> list[float]: ...

    @overload
    def get_float_list(
        self,
        section: LiteralString,
        key: LiteralString,
        *,
        fallback: Sequence[float],
        separator: str = ",",
    ) -> list[float]: ...

    def get_float_list(
        self,
        section: LiteralString,
        key: LiteralString,
        *,
        separator: str = ",",
        **kwargs: Sequence[float],
    ) -> list[float]:
        return get_float_list(self, self._sample_name, section, key, separator=separator, **kwargs)

    @overload
    def get_decimal_list(
        self,
        section: LiteralString,
        key: LiteralString,
        *,
        separator: str = ",",
    ) -> list[Decimal]: ...

    @overload
    def get_decimal_list(
        self,
        section: LiteralString,
        key: LiteralString,
        *,
        fallback: Sequence[float],
        separator: str = ",",
    ) -> list[Decimal]: ...

    def get_decimal_list(
        self,
        section: LiteralString,
        key: LiteralString,
        *,
        separator: str = ",",
        **kwargs: Sequence[float],
    ) -> list[Decimal]:
        return get_decimal_list(self, self._sample_name, section, key, separator=separator, **kwargs)

    @overload
    def get_slice_sequence(
        self,
        section: LiteralString,
        key: LiteralString,
        *,
        slice_separator: str | Sequence[str] = ("..", ":"),
        items_separator: str | Sequence[str] = (",", ";"),
    ) -> SliceSequence: ...

    @overload
    def get_slice_sequence(
        self,
        section: LiteralString,
        key: LiteralString,
        *,
        fallback: SliceSequence,
        slice_separator: str | Sequence[str] = ("..", ":"),
        items_separator: str | Sequence[str] = (",", ";"),
    ) -> SliceSequence: ...

    def get_slice_sequence(
        self,
        section: LiteralString,
        key: LiteralString,
        *,
        slice_separator: str | Sequence[str] = ("..", ":"),
        items_separator: str | Sequence[str] = (",", ";"),
        **kwargs: SliceSequence,
    ) -> SliceSequence:
        return get_slice_sequence(
            self,
            self._sample_name,
            section,
            key,
            slice_separator=slice_separator,
            items_separator=items_separator,
            **kwargs,
        )
