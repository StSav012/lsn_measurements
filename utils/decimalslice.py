# -*- coding: utf-8 -*-
from decimal import Decimal
from typing import Final, Iterator

__all__ = ["DecimalSlice"]


class DecimalSlice:
    def __init__(self, start: Decimal, stop: Decimal, step: Decimal = 1.0) -> None:
        if step == 0.0 and start != stop:
            raise ValueError
        if (step > 0.0) == (start > stop):
            start, stop = stop, start
        self._start: Final[Decimal] = start
        self._stop: Final[Decimal] = stop
        self._step: Final[Decimal] = step
        self._min: Final[Decimal] = Decimal(min(self.start, self.stop))
        self._max: Final[Decimal] = Decimal(max(self.start, self.stop))

    @property
    def start(self) -> Decimal:
        return self._start

    @property
    def stop(self) -> Decimal:
        return self._stop

    @property
    def step(self) -> Decimal:
        return self._step

    @property
    def min(self) -> Decimal:
        return self._min

    @property
    def max(self) -> Decimal:
        return self._max

    def contains(self, value: Decimal) -> bool:
        return self.min <= value <= self.max

    def __iter__(self) -> Iterator[Decimal]:
        v: Decimal = self.start
        while self.contains(v):
            yield v
            v += self.step
            if self.step == 0.0:
                break

    def __getitem__(self, item: int) -> Decimal:
        return self.start + item * self.step

    def __len__(self) -> int:
        if self.step == 0.0:
            return 1
        return int((self.stop - self.start) // self.step) + 1

    def __str__(self) -> str:
        return f"{self.start} to {self.stop} by {self.step}"

    @staticmethod
    def from_string(text: str) -> "DecimalSlice":
        start: Decimal
        stop: Decimal
        step: Decimal
        if ".." not in text:
            start = stop = Decimal(text)
            step = Decimal(0.0)
        elif text.count("..") == 1:
            start, stop = map(Decimal, text.split("..", maxsplit=1))
            step = Decimal(1.0)
        elif text.count("..") == 2:
            start, step, stop = map(Decimal, text.split("..", maxsplit=2))
        else:
            raise ValueError
        return DecimalSlice(start, stop, step)
