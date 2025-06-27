import numpy as np

__all__ = ["Count"]


class Count:
    def __init__(self) -> None:
        self._count: int = 0
        self._last_value: int = 0

        self.payload: tuple[float, float, float] = (np.nan, np.nan, np.nan)
        self.loaded: bool = False
        self.loadable: bool = True

    def inc(self, step: int = 1) -> None:
        self._count += int(step)

    def reset(self) -> None:
        self._last_value = self._count
        self._count = 0

    def __int__(self) -> int:
        return self._count

    def __ceil__(self) -> int:
        return self._last_value
