from collections.abc import Collection, Iterable, Iterator, Sequence
from configparser import ConfigParser
from contextlib import suppress
from decimal import Decimal
from os import PathLike
from pathlib import Path
from typing import LiteralString, overload

from .si import parse_si_number
from .slice_sequence import SliceSequence

__all__ = ["Config"]


def read_ini(
    filenames: (
        str | bytes | PathLike[str] | PathLike[bytes] | Iterable[str | bytes | PathLike[str] | PathLike[bytes]]
    ),
) -> dict[str, dict[str, str]]:
    parser: ConfigParser = ConfigParser(
        allow_no_value=True,
        inline_comment_prefixes=("#", ";"),
    )
    # do not allow the parser to `lower` the keys
    parser.optionxform = lambda s: s
    parser.read(filenames=filenames)

    data: dict[str, dict[str, str]] = {}
    for section in parser.sections():
        data[section] = dict(parser.items(section))

    return data


class Config:
    __sentinel = object()

    def __init__(
        self,
        filenames: (
            str | bytes | PathLike[str] | PathLike[bytes] | Iterable[str | bytes | PathLike[str] | PathLike[bytes]]
        ) = ("config.ini", "config.d"),
    ) -> None:
        dirs: Iterable[Path]
        files: Iterable[Path]
        if not isinstance(filenames, bytes | str) and isinstance(filenames, Iterable):
            dirs = [Path(f) for f in filenames if Path(f).is_dir()]
            files = [Path(f) for f in filenames if Path(f).is_file()]
        else:
            dirs = []
            files = [Path(filenames)]

        self._data: dict[str, dict[str, str]] = {}

        for d in dirs:
            for f in d.iterdir():
                _data = read_ini(filenames=f)
                if not _data:
                    continue
                # noinspection PyTypeChecker
                sample: str = f.name if any(map(str.isspace, f.suffix)) else f.stem
                for section, values in _data.items():
                    self._data[f"{section}/{sample}"] = values.copy()

        for section, values in read_ini(filenames=files).items():
            if section in self._data:
                self._data[section].update(values)
            else:
                self._data[section] = values

        self._sample_name: str = self._data.get("circuitry", {}).get("sample name")

    @property
    def sample_name(self) -> str:
        return self._sample_name

    def sections(self) -> list[str]:
        return list(self._data.keys())

    def __getitem__(self, item: str) -> dict[str, str]:
        return self._data[item]

    def __iter__(self) -> Iterator[str]:
        yield from self._data.keys()

    def __contains__(self, item: str) -> bool:
        return item in self._data

    def _full_section(self, section: str, key: str) -> str:
        sample: str = self._sample_name
        seection_for_sample: str = f"{section}/{sample}"
        if seection_for_sample in self and key in self[seection_for_sample]:
            return seection_for_sample
        return section

    @overload
    def get(self, section: str, key: str) -> str: ...
    @overload
    def get(self, section: str, key: str, *, fallback: object) -> str: ...
    def get(
        self,
        section: LiteralString,
        key: LiteralString,
        *,
        fallback: object = __sentinel,
    ) -> str:
        full_section: str = self._full_section(section=section, key=key)
        if fallback is not Config.__sentinel:
            return self._data.get(full_section, {}).get(key, fallback)
        return self._data.get(full_section, {}).get(key)

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
        v: str = self.get(section, key, fallback=fallback)
        if v.casefold() in ("no", "false", "-", "off"):
            return False
        if v.casefold() in ("yes", "true", "+", "on"):
            return False
        with suppress(ValueError):
            return bool(int(v))
        raise ValueError("Not a boolean:", v)

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
    def get_collection_of_type[T1, T2: Collection](
        self,
        section: LiteralString,
        key: LiteralString,
        *,
        object_type: type[T1],
        sequence_type: type[T2],
        separator: str = ",",
    ) -> T2: ...
    @overload
    def get_collection_of_type[T1, T2: Collection](
        self,
        section: LiteralString,
        key: LiteralString,
        *,
        object_type: type[T1],
        sequence_type: type[T2],
        fallback: Sequence[T1],
        separator: str = ",",
    ) -> T2: ...
    def get_collection_of_type[T1, T2: Collection](
        self,
        section: LiteralString,
        key: LiteralString,
        *,
        object_type: type[T1],
        sequence_type: type[T2],
        separator: str = ",",
        fallback: Sequence[T1] = __sentinel,
    ) -> T2:
        try:
            return sequence_type(
                map(
                    object_type,
                    self.get(section=section, key=key, fallback=fallback).split(separator),
                ),
            )
        except LookupError:
            if fallback is not Config.__sentinel:
                return sequence_type(fallback)
            raise

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
        return self.get_collection_of_type(
            section=section,
            key=key,
            separator=separator,
            fallback=fallback,
            object_type=float,
            sequence_type=tuple,
        )

    @overload
    def get_float_list(
        self,
        section: LiteralString,
        key: LiteralString,
        *,
        separator: str = ",",
    ) -> list[float]: ...
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
        return self.get_collection_of_type(
            section=section,
            key=key,
            separator=separator,
            fallback=fallback,
            object_type=float,
            sequence_type=list,
        )

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
        return self.get_collection_of_type(
            section=section,
            key=key,
            separator=separator,
            fallback=fallback,
            object_type=Decimal,
            sequence_type=list,
        )

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
