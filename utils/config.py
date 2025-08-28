from collections.abc import Iterable, Sequence
from configparser import ConfigParser
from decimal import Decimal
from os import PathLike
from typing import LiteralString, overload

from .si import parse_si_number
from .slice_sequence import SliceSequence

__all__ = ["Config"]


class Config(ConfigParser):
    __sentinel = object()

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

        self._sample_name: str = super().get(section="circuitry", option="sample name")

    @property
    def sample_name(self) -> str:
        return self._sample_name

    def _full_section(self, section: str, key: str) -> str:
        sample: str = self._sample_name
        if f"{section}/{sample}" in self.sections() and key in self[f"{section}/{sample}"]:
            return f"{section}/{sample}"
        return section

    # noinspection PyMethodOverriding
    @overload
    def get(self, section: str, key: str) -> str: ...
    # noinspection PyMethodOverriding
    @overload
    def get(self, section: str, key: str, *, fallback: object) -> str: ...
    # noinspection PyMethodOverriding
    def get(
        self,
        section: LiteralString,
        key: LiteralString,
        *,
        fallback: object = __sentinel,
    ) -> str:
        full_section: str = self._full_section(section=section, key=key)
        if fallback is not Config.__sentinel:
            return super().get(full_section, key, fallback=fallback)
        return super().get(full_section, key)

    @overload
    def get_str(self, section: LiteralString, key: LiteralString) -> str: ...

    @overload
    def get_str(self, section: LiteralString, key: LiteralString, *, fallback: str) -> str: ...

    def get_str(self, section: LiteralString, key: LiteralString, *, fallback: str = __sentinel) -> str:
        return self.get(section=section, key=key, fallback=fallback)

    @overload
    def get_bool(self, section: LiteralString, key: LiteralString) -> bool: ...

    @overload
    def get_bool(self, section: LiteralString, key: LiteralString, *, fallback: bool) -> bool: ...

    def get_bool(self, section: LiteralString, key: LiteralString, *, fallback: bool = __sentinel) -> bool:
        full_section: str = self._full_section(section=section, key=key)
        if fallback is not Config.__sentinel:
            return self.getboolean(full_section, key, fallback=fallback)
        return self.getboolean(full_section, key)

    @overload
    def get_int(self, section: LiteralString, key: LiteralString) -> int: ...

    @overload
    def get_int(self, section: LiteralString, key: LiteralString, *, fallback: int) -> int: ...

    def get_int(self, section: LiteralString, key: LiteralString, *, fallback: int = __sentinel) -> int:
        return int(self.get(section=section, key=key, fallback=fallback))

    @overload
    def get_float(self, section: LiteralString, key: LiteralString) -> float: ...

    @overload
    def get_float(self, section: LiteralString, key: LiteralString, *, fallback: float) -> float: ...

    def get_float(self, section: LiteralString, key: LiteralString, *, fallback: float = __sentinel) -> float:
        return parse_si_number(self.get(section=section, key=key, fallback=fallback))

    @overload
    def get_decimal(self, section: LiteralString, key: LiteralString) -> Decimal: ...

    @overload
    def get_decimal(self, section: LiteralString, key: LiteralString, *, fallback: float) -> Decimal: ...

    def get_decimal(self, section: LiteralString, key: LiteralString, *, fallback: float = __sentinel) -> Decimal:
        return Decimal(self.get(section=section, key=key, fallback=fallback))

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
        fallback: Sequence[float] = __sentinel,
    ) -> tuple[float, ...]:
        try:
            return tuple(
                map(
                    float,
                    self.get(section=section, key=key, fallback=fallback).split(separator),
                ),
            )
        except LookupError:
            if fallback is not Config.__sentinel:
                return tuple(fallback)
            raise

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
        fallback: Sequence[float] = __sentinel,
    ) -> list[float]:
        try:
            return list(
                map(
                    float,
                    self.get(section=section, key=key, fallback=fallback).split(separator),
                ),
            )
        except LookupError:
            if fallback is not Config.__sentinel:
                return list(fallback)
            raise

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
        fallback: Sequence[float] = __sentinel,
    ) -> list[Decimal]:
        try:
            return list(
                map(
                    Decimal,
                    self.get(section=section, key=key, fallback=fallback).split(separator),
                ),
            )
        except LookupError:
            if fallback is not Config.__sentinel:
                return list(map(Decimal, fallback))
            raise

    @overload
    def get_slice_sequence(
        self,
        section: LiteralString,
        key: LiteralString,
        *,
        slice_separator: str | Iterable[str] = ("..", ":"),
        items_separator: str | Iterable[str] = (",", ";"),
    ) -> SliceSequence: ...

    @overload
    def get_slice_sequence(
        self,
        section: LiteralString,
        key: LiteralString,
        *,
        fallback: SliceSequence,
        slice_separator: str | Iterable[str] = ("..", ":"),
        items_separator: str | Iterable[str] = (",", ";"),
    ) -> SliceSequence: ...

    def get_slice_sequence(
        self,
        section: LiteralString,
        key: LiteralString,
        *,
        slice_separator: str | Iterable[str] = ("..", ":"),
        items_separator: str | Iterable[str] = (",", ";"),
        fallback: SliceSequence = __sentinel,
    ) -> SliceSequence:
        return SliceSequence(
            self.get(section=section, key=key, fallback=fallback),
            slice_separator=slice_separator,
            items_separator=items_separator,
        )
