from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "half_sine_segment",
    "half_sine_segments",
    "linear_segment",
    "linear_segments",
    "parabolic_segment",
    "quarter_sine_segments",
]


def linear_segment(
    start_point: float,
    end_point: float,
    points_count: int,
    *,
    endpoint: bool = True,
) -> NDArray[np.float64]:
    return np.linspace(start_point, end_point, points_count, endpoint=endpoint, dtype=np.float64)


def half_sine_segment(
    start_point: float,
    end_point: float,
    points_count: int,
    *,
    endpoint: bool = True,
) -> NDArray[np.float64]:
    return start_point + (end_point - start_point) * 0.5 * (
        1.0 - np.cos(np.linspace(0.0, np.pi, points_count, endpoint=endpoint))
    )


def quarter_sine_segment(
    start_point: float,
    end_point: float,
    points_count: int,
    *,
    endpoint: bool = True,
) -> NDArray[np.float64]:
    return start_point + (end_point - start_point) * (
        np.sin(np.linspace(0.0, 0.5 * np.pi, points_count, endpoint=endpoint))
    )


def parabolic_segment(
    start_point: float,
    end_point: float,
    points_count: int,
    *,
    endpoint: bool = True,
) -> NDArray[np.float64]:
    if start_point <= 0.0 and end_point <= 0.0:
        return -np.square(
            np.linspace(
                np.sqrt(-start_point),
                np.sqrt(-end_point),
                points_count,
                endpoint=endpoint,
            ),
        )
    if start_point <= 0.0 <= end_point:
        return np.concatenate(
            (
                -np.square(
                    np.linspace(
                        np.sqrt(-start_point),
                        0.0,
                        int(np.floor(points_count * abs(-start_point / (end_point - start_point)))),
                        endpoint=False,
                    ),
                ),
                np.square(
                    np.linspace(
                        0.0,
                        np.sqrt(end_point),
                        int(np.ceil(points_count * abs(end_point / (end_point - start_point)))),
                        endpoint=endpoint,
                    ),
                ),
            ),
        )
    if start_point >= 0.0 >= end_point:
        return np.concatenate(
            (
                np.square(
                    np.linspace(
                        np.sqrt(start_point),
                        0.0,
                        int(np.floor(points_count * abs(start_point / (end_point - start_point)))),
                        endpoint=False,
                    ),
                ),
                -np.square(
                    np.linspace(
                        0.0,
                        np.sqrt(-end_point),
                        int(np.ceil(points_count * abs(end_point / (end_point - start_point)))),
                        endpoint=endpoint,
                    ),
                ),
            ),
        )
    return np.square(
        np.linspace(
            np.sqrt(start_point),
            np.sqrt(end_point),
            points_count,
            endpoint=endpoint,
        ),
    )


def linear_segments(
    current_values: Sequence[float],
    points_count: int,
) -> NDArray[np.float64]:
    if len(current_values) < 2:
        raise ValueError
    if len(current_values) > points_count - 1:
        raise ValueError
    points_per_part: int = (points_count - 1) // (len(current_values) - 1)
    return np.concatenate(
        (
            np.concatenate(
                [
                    linear_segment(prev_point, next_point, points_per_part, endpoint=False)
                    for prev_point, next_point in zip(current_values, current_values[1:], strict=False)
                ],
            ),
            [current_values[-1]] * ((points_count - 1) % (len(current_values) - 1)),
            [current_values[-1]],
        ),
        dtype=np.float64,
    )


def half_sine_segments(
    current_values: Sequence[float],
    points_count: int,
) -> NDArray[np.float64]:
    if len(current_values) < 2:
        raise ValueError
    if len(current_values) > points_count - 1:
        raise ValueError
    points_per_part: int = (points_count - 1) // (len(current_values) - 1)
    return np.concatenate(
        (
            np.concatenate(
                [
                    half_sine_segment(prev_point, next_point, points_per_part, endpoint=False)
                    for prev_point, next_point in zip(current_values, current_values[1:], strict=False)
                ],
            ),
            [current_values[-1]] * ((points_count - 1) % (len(current_values) - 1)),
            [current_values[-1]],
        ),
        dtype=np.float64,
    )


def quarter_sine_segments(
    current_values: Sequence[float],
    points_count: int,
) -> NDArray[np.float64]:
    if len(current_values) < 2:
        raise ValueError
    if len(current_values) > points_count - 1:
        raise ValueError
    points_per_part: int = (points_count - 1) // (len(current_values) - 1)
    return np.concatenate(
        (
            np.concatenate(
                [
                    quarter_sine_segment(prev_point, next_point, points_per_part, endpoint=False)
                    for prev_point, next_point in zip(current_values, current_values[1:], strict=False)
                ],
            ),
            [current_values[-1]] * ((points_count - 1) % (len(current_values) - 1)),
            [current_values[-1]],
        ),
        dtype=np.float64,
    )
