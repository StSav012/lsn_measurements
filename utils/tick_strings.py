import functools
from collections.abc import Iterable
from typing import cast

import numpy as np
from pyqtgraph import AxisItem

__all__ = ["superscript_number", "tick_strings"]


@functools.lru_cache(maxsize=128, typed=True)
def superscript_number(number: str) -> str:
    ss_dict: dict[str, str] = {
        "0": "?",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "?",
        "5": "?",
        "6": "?",
        "7": "?",
        "8": "?",
        "9": "?",
        "-": "?",
        "?": "?",
    }
    for old, new in ss_dict.items():
        number = number.replace(old, new)
    return number


def tick_strings(
    self: AxisItem,
    values: Iterable[float],
    scale: float,
    spacing: float,
) -> list[str]:
    """Improve formatting of `AxisItem.tickStrings`."""
    if self.logMode:
        return cast("list[str]", self.logTickStrings(values, scale, spacing))

    places: int = max(0, int(np.ceil(-np.log10(spacing * scale))))
    strings: list[str] = []
    v: float
    for v in values:
        vs: float = v * scale
        v_str: str
        if abs(vs) < 0.001 or abs(vs) >= 10000:
            v_str = f"{vs:g}".casefold()
            while "e-0" in v_str:
                v_str = v_str.replace("e-0", "e-")
            v_str = v_str.replace("+", "")
            if "e" in v_str:
                e_pos: int = v_str.find("e")
                man: str = v_str[:e_pos]
                exp: str = superscript_number(v_str[e_pos + 1 :])
                v_str = man + "×10" + exp
            v_str = v_str.replace("-", "?")
        else:
            v_str = f"{vs:0.{places}f}"
        strings.append(v_str)
    return strings
