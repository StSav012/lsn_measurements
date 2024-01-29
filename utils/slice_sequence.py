# coding: utf-8
from __future__ import annotations

import math
from typing import Iterator, Self, Sequence

from .si import parse_si_number
from .string_utils import multi_split, nth_occurrence

__all__ = ["SliceSequence"]


def float_range(
    start: float,
    stop: float,
    step: float = 1.0,
    *,
    rel_tol: float = 1e-09,
    abs_tol: float = 0.0,
) -> list[float]:
    values: list[float] = [start + index * step for index in range(int((stop - start) // step) + 1)]
    if not values:
        return values
    if math.isclose(values[-1] + step, stop, rel_tol=rel_tol, abs_tol=abs_tol):
        values.append(stop)
    return values


class SliceSequence:
    def __init__(
        self,
        text: str | Self = "",
        *,
        slice_separator: str | Sequence[str] = ("..", ":"),
        items_separator: str | Sequence[str] = (",", ";"),
    ) -> None:
        self._slice_separators: tuple[str] = (
            (slice_separator,) if isinstance(slice_separator, str) else tuple(slice_separator)
        )
        self._items_separators: tuple[str] = (
            (items_separator,) if isinstance(items_separator, str) else tuple(items_separator)
        )

        self._items: list[float] = list(text) if isinstance(text, SliceSequence) else self._parse(text)

    def _parse(self, text: str) -> list[float]:
        def _parse_slice(slice_text: str) -> list[float]:
            if not slice_text:
                return []
            slice_text = slice_text.strip()
            parts: list[str] = multi_split(slice_text, self._slice_separators)
            _slice: list[float] = list(map(parse_si_number, parts))
            if len(_slice) == 1:
                return _slice
            elif len(_slice) == 2 and not any(map(math.isnan, _slice)):
                return float_range(_slice[0], _slice[-1])
            elif len(_slice) == 3 and not any(map(math.isnan, _slice)):
                return float_range(_slice[0], _slice[-1], step=_slice[1])
            else:
                error_text: str = f"Invalid slice notation: {slice_text}"
                if any(map(math.isnan, _slice)):
                    nan_index: int = _slice.index(math.nan)
                    line_length: int = 29 + nth_occurrence(slice_text, self._slice_separators, nan_index)
                    if parts[nan_index]:
                        line_length += 1 + len(parts[nan_index]) // 2
                    error_text += "\n" + "-" * line_length + " here -^"
                elif len(_slice) > 3:
                    error_text += "\n" + "-" * (28 + len(slice_text)) + " here -^"
                raise ValueError(error_text)

        values: list[float] = []
        word: str
        for word in multi_split(text, self._items_separators):
            values.extend(_parse_slice(word))
        return values

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._items})"

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, item: int) -> float:
        try:
            return self._items[item]
        except IndexError:
            raise IndexError(f"item number {item} does not exist among {self._items}")

    def __iter__(self) -> Iterator[float]:
        yield from self._items

    def __format__(self, format_spec: str) -> str:
        if not self._items:
            return "[]"
        if len(self._items) == 1:
            return f"{self._items[0]:{format_spec}}"
        return "[" + ", ".join(f"{item:{format_spec}}" for item in self._items) + "]"

    def __bool__(self) -> bool:
        return bool(self._items)


if __name__ == "__main__":
    # print(multi_split('1,2;3 ,4, 5,', (',', ';')))
    # print(SliceSequence('1,2;3 ,4, 5'))
    # print(SliceSequence('1:2;3:0.2:4, 5'))
    # print(f"{SliceSequence('1:2;3:0.2:4, 5'):.6f}")
    # print(f"{SliceSequence('1;2,'):8.4f}")
    # print(si_prefix('mK'), si_prefix('dam'))
    # print(f"{SliceSequence('1:2;3:80m:4, 5K')}")
    # print(f"{SliceSequence('1:2;3:80m:4:, 5K')}")
    # SliceSequence('1:2;3:80m:4, 1:abc:2')
    print(SliceSequence("1:2;3:50m:4, 5k"))
    print(SliceSequence(SliceSequence("1:2;3:50m:4, 5k")))
    print(4 in SliceSequence("1:2;3:50m:4, 5k"))
