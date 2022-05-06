# -*- coding: utf-8 -*-
from numbers import Real
from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate


def attenuation(attenuation_voltage: Union[Real, NDArray[Real]], cal_source: str = 'psi'):
    cal_text_psi: str = """\
0	8.9
0.4	8.6
0.5	7.9
0.6	7.1
0.7	6.1
0.8	5
0.9	4
1	3.1
1.1	2.2
1.2	1.3
1.3	0.4
1.4	-0.4
1.5	-1.2
1.6	-2
1.7	-2.7
1.8	-3.4
1.9	-4
2	-4.6
2.1	-5.2
2.2	-5.8
2.3	-6.3
2.4	-7.1
2.5	-7.6
2.6	-8
2.7	-8.5
2.8	-8.9
2.9	-9.4
3	-9.8
3.1	-10.2
3.2	-10.5
3.3	-10.9
3.4	-11.2
3.5	-11.6
3.6	-11.9
3.7	-12.2
3.8	-12.5
3.9	-12.8
4	-13.1
4.1	-13.3
4.2	-13.5
4.3	-13.8
4.4	-14
4.5	-14.3
4.6	-14.6
4.7	-14.8
4.8	-15
4.9	-15.3
5	-15.5
5.5	-16.5
6	-17.4
6.5	-18.2
7	-19
7.5	-19.7
8	-20.3
8.5	-20.9
9	-21.4
9.5	-21.9
10	-22.4
"""

    cal_text_paraffin: str = """\
0	-1.48
0.25	-1.5
0.3	-1.53
0.32	-1.55
0.34	-1.59
0.36	-1.64
0.38	-1.7
0.4	-1.78
0.45	-2.05
0.55	-2.82
0.65	-3.77
0.75	-4.8
0.85	-5.85
0.95	-6.9
1.05	-7.9
1.15	-8.9
1.25	-9.82
1.35	-10.7
1.45	-11.55
1.55	-12.35
1.65	-13.1
1.75	-13.83
1.85	-14.51
1.95	-15.16
2.05	-15.78
2.15	-16.36
2.25	-16.92
2.35	-17.44
2.45	-17.95
2.55	-18.44
2.65	-18.9
2.75	-19.3
2.85	-19.77
2.95	-20.18
3.05	-20.58
3.15	-20.95
3.25	-21.32
3.35	-21.68
3.45	-22
3.55	-22.34
3.65	-22.68
3.75	-22.97
3.85	-23.27
3.95	-23.55
4.05	-23.85
4.55	-25.13
5.05	-26.26
5.55	-27.25
6.05	-28.15
6.55	-28.94
7.05	-29.68
7.75	-30.6
8.75	-31.74
9.75	-32.7
"""
    if cal_source.casefold() == 'psi'.casefold():
        cal_text: str = cal_text_psi
    elif cal_source.casefold() == 'paraffin'.casefold():
        cal_text: str = cal_text_paraffin
    else:
        raise ValueError('Invalid calibration source:', cal_source)
    cal_data: np.ndarray = np.array(
        [[float(i) for i in line.split()[:2]] for line in cal_text.strip().splitlines() if line.strip()]).T
    cal_x = cal_data[0]
    cal_y = cal_data[1]
    return interpolate.interp1d(cal_x, cal_y, kind='quadratic')(attenuation_voltage) - cal_y[0]
