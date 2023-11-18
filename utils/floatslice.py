# -*- coding: utf-8 -*-
from typing import Final, Iterator

__all__ = ["FloatSlice"]


class FloatSlice:
    def __init__(self, start: float, stop: float, step: float = 1.0) -> None:
        if step == 0.0 and start != stop:
            raise ValueError
        if (step > 0.0) == (start > stop):
            start, stop = stop, start
        self._start: Final[float] = start
        self._stop: Final[float] = stop
        self._step: Final[float] = step
        self._min: Final[float] = min(self.start, self.stop)
        self._max: Final[float] = max(self.start, self.stop)

    @property
    def start(self) -> float:
        return self._start

    @property
    def stop(self) -> float:
        return self._stop

    @property
    def step(self) -> float:
        return self._step

    @property
    def min(self) -> float:
        return self._min

    @property
    def max(self) -> float:
        return self._max

    def contains(self, value: float, eps: float = 1e-8) -> bool:
        return -eps * abs(self.min) + self.min <= value <= eps * abs(self.max) + self.max

    def __iter__(self) -> Iterator[float]:
        v: float = self.start
        while self.contains(v):
            yield v
            v += self.step
            if self.step == 0.0:
                break

    def __getitem__(self, item: int) -> float:
        return self.start + item * self.step

    def __len__(self) -> int:
        if self.step == 0.0:
            return 1
        return int((self.stop - self.start) // self.step) + 1

    def __str__(self) -> str:
        return f"{self.start} to {self.stop} by {self.step}"

    @staticmethod
    def from_string(text: str) -> "FloatSlice":
        start: float
        stop: float
        step: float
        if ".." not in text:
            start = stop = float(text)
            step = 0.0
        elif text.count("..") == 1:
            start, stop = map(float, text.split("..", maxsplit=1))
            step = 1.0
        elif text.count("..") == 2:
            start, step, stop = map(float, text.split("..", maxsplit=2))
        else:
            raise ValueError
        return FloatSlice(start, stop, step)
