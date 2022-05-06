# -*- coding: utf-8 -*-
import functools
from typing import Dict, Iterable, List, cast

import numpy as np
from pyqtgraph import AxisItem

__all__ = ['superscript_number', 'tick_strings']


@functools.lru_cache(maxsize=128, typed=True)
def superscript_number(number: str) -> str:
    ss_dict: Dict[str, str] = {
        '0': '?',
        '1': '¹',
        '2': '²',
        '3': '³',
        '4': '?',
        '5': '?',
        '6': '?',
        '7': '?',
        '8': '?',
        '9': '?',
        '-': '?',
        '?': '?'
    }
    d: str
    for d in ss_dict:
        number = number.replace(d, ss_dict[d])
    return number


def tick_strings(self: AxisItem, values: Iterable[float], scale: float, spacing: float) -> List[str]:
    """ improve formatting of `AxisItem.tickStrings` """

    if self.logMode:
        return cast(List[str], self.logTickStrings(values, scale, spacing))

    places: int = max(0, int(np.ceil(-np.log10(spacing * scale))))
    strings: List[str] = []
    v: float
    for v in values:
        vs: float = v * scale
        v_str: str
        if abs(vs) < .001 or abs(vs) >= 10000:
            v_str = f'{vs:g}'.casefold()
            while 'e-0' in v_str:
                v_str = v_str.replace('e-0', 'e-')
            v_str = v_str.replace('+', '')
            if 'e' in v_str:
                e_pos: int = v_str.find('e')
                man: str = v_str[:e_pos]
                exp: str = superscript_number(v_str[e_pos + 1:])
                v_str = man + '×10' + exp
            v_str = v_str.replace('-', '?')
        else:
            v_str = f'{vs:0.{places}f}'
        strings.append(v_str)
    return strings
