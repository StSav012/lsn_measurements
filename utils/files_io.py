# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, SupportsFloat, SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["load_txt", "save_txt", "read_340_table"]


def is_float(val: Any) -> bool:
    if not isinstance(val, (SupportsFloat, SupportsIndex, str, bytes, bytearray)):
        return False
    try:
        float(val)
    except ValueError:
        return False
    else:
        return True


def save_txt(
    filename: str | Path,
    x: ArrayLike,
    fmt: str | Iterable[str] = "%.18e",
    delimiter: str = " ",
    newline: str = os.linesep,
    header: str = "",
    footer: str = "",
    comments: str = "# ",
    encoding: str | None = "utf-8",
) -> None:
    """
    from `numpy.savetxt`
    Save an array to a text file.

    Parameters
    ----------
    filename : {pathlib.Path}
        If the filename ends in ``.gz``, the file is automatically saved in
        compressed gzip format.  `loadtxt` understands gzipped files
        transparently.
    x : {ArrayLike}
        1D or 2D array_like
        Data to be saved to a text file.
    fmt : str or sequence of strs, optional
        A single format (%10.5f), a sequence of formats, or a
        multi-format string, e.g. 'Iteration %d -- %10.5f', in which
        case `delimiter` is ignored. For complex `X`, the legal options
        for `fmt` are:
        * a single specifier, `fmt='%.4e'`, resulting in numbers formatted
          like `' (%s+%sj)' % (fmt, fmt)`
        * a full string specifying every real and imaginary part, e.g.
          `' %.4e %+.4ej %.4e %+.4ej %.4e %+.4ej'` for 3 columns
        * a list of specifiers, one per column - in this case, the real
          and imaginary part must have separate specifiers,
          e.g. `['%.3e + %.3ej', '(%.15e%+.15ej)']` for 2 columns
    delimiter : str, optional
        String or character separating columns.
    newline : str, optional
        String or character separating lines.
    header : str, optional
        String that will be written at the beginning of the file.
    footer : str, optional
        String that will be written at the end of the file.
    comments : str, optional
        String that will be prepended to the ``header`` and ``footer`` strings,
        to mark them as comments. Default: '# ',  as expected by e.g.
        ``numpy.loadtxt``.
    encoding : {None, str}, optional
        Encoding used to encode the outputfile. Does not apply to output
        streams. If the encoding is something other than 'bytes' or 'latin1'
        you will not be able to load the file in NumPy versions < 1.14. Default
        is 'latin1'.

    Notes
    -----
    Further explanation of the `fmt` parameter
    (``%[flag]width[.precision]specifier``):
    flags:
        * ``-`` : left justify
        * ``+`` : Forces to precede result with + or -.
        * ``0`` : Left pad the number with zeros instead of space (see width).
    width:
        Minimum number of characters to be printed. The value is not truncated
        if it has more characters.
    precision:
        - For integer specifiers (eg. ``d,i,o,x``), the minimum number of
          digits.
        - For ``e, E`` and ``f`` specifiers, the number of digits to print
          after the decimal point.
        - For ``g`` and ``G``, the maximum number of significant digits.
        - For ``s``, the maximum number of characters.
    specifiers:
        ``c`` : character
        ``d`` or ``i`` : signed decimal integer
        ``e`` or ``E`` : scientific notation with ``e`` or ``E``.
        ``f`` : decimal floating point
        ``g,G`` : use the shorter of ``e,E`` or ``f``
        ``o`` : signed octal
        ``s`` : string of characters
        ``u`` : unsigned decimal integer
        ``x,X`` : unsigned hexadecimal integer
    This explanation of ``fmt`` is not complete, for an exhaustive
    specification see [1]_.

    References
    ----------
    .. [1] `Format Specification Mini-Language
           <https://docs.python.org/library/string.html#format-specification-mini-language>`_,
           Python Documentation.
    """

    try:
        x = np.asarray(x)

        # Handle 1-dimensional arrays
        if x.ndim == 0 or x.ndim > 2:
            raise ValueError("Expected 1D or 2D array, got %dD array instead" % x.ndim)
        elif x.ndim == 1:
            # Common case -- 1d array of numbers
            if x.dtype.names is None:
                x = np.atleast_2d(x).T
                ncol = 1

            # Complex dtype -- each field indicates a separate column
            else:
                ncol = len(x.dtype.names)
        else:
            ncol = x.shape[1]

        # `fmt` can be a string with multiple insertion points or a
        # list of formats.  E.g. '%10.5f\t%10d' or ('%10.5f', '$10d')
        if type(fmt) in (list, tuple):
            if len(fmt) != ncol:
                raise AttributeError(f"fmt has wrong shape. {fmt}")
            fmt = delimiter.join(fmt)
        elif isinstance(fmt, str):
            n_fmt_chars = fmt.count("%")
            if n_fmt_chars == 1:
                fmt = delimiter.join([fmt] * ncol)
            elif n_fmt_chars != ncol:
                raise ValueError(f"fmt has wrong number of %% formats: {fmt}")
        else:
            raise ValueError(f"invalid format: {fmt!r}")

        if isinstance(filename, str):
            filename = Path(filename)
        with filename.open("wt", encoding=encoding, newline=newline) as fh:
            if header:
                header = header.replace("\n", "\n" + comments)
                fh.write(comments + header + "\n")
            row_pack: int = 100000
            row_pack_fmt: str
            row_pack_fmt = "\n".join([fmt] * row_pack) + "\n"
            for row in range(0, x.shape[0] - row_pack, row_pack):
                try:
                    fh.write(row_pack_fmt % tuple(x[row : row + row_pack, ...].ravel()))
                except TypeError:
                    raise TypeError("Mismatch between array data type and format specifier")
            row_pack = x.shape[0] % row_pack
            row_pack_fmt = "\n".join([fmt] * row_pack) + "\n"
            try:
                fh.write(row_pack_fmt % tuple(x[-row_pack:, ...].ravel()))
            except TypeError:
                raise TypeError("Mismatch between array data type and format specifier")

            if footer:
                footer = footer.replace("\n", "\n" + comments)
                fh.write(comments + footer + "\n")
    finally:
        pass


def load_txt(
    filename: str | Path,
    sep: str | None = None,
    encoding: str | None = None,
    errors: str | None = None,
) -> tuple[NDArray[float], tuple[str]]:
    """
    Load data from a text file, possibly with a header.

    Parameters
    ----------
    filename : {str, pathlib.Path}
        Name of the file to read text data from.
    sep : {None, str}, optional
        The delimiter between the values as they are written in the file.
        None (the default value) means split according to any whitespace,
            and discard empty strings from the result.
    encoding : {None, str}, optional
        Encoding used to encode the input file. See [1]_ for the available values.
    errors : {None, str}, optional
        Way of handling encoding errors. See [1]_ for the available values.

    References
    ----------
    .. [1] `Python Built-In Functions: open
           <https://docs.python.org/3/library/functions.html#open>`_,
           Python Documentation.

    Returns
    -------
    data : numpy.typing.NDArray[float]
        An array object containing the numerical data read from the file.
    titles : tuple[str]
        The first line of the file split by `sep`.
    """

    filename = Path(filename)
    data: list[list[str]] = [
        word.split(sep=sep) for word in filename.read_text(encoding=encoding, errors=errors).splitlines() if word
    ]
    if not data:
        return np.empty(0, float), tuple()
    header: tuple[str]
    numeric_data: NDArray[float]
    if all(is_float(word) for word in data[0]):
        header = tuple()
        numeric_data = np.array(
            [[float(word) for word in line] for line in data],
            dtype=float,
        )
    else:
        header = tuple(data[0])
        numeric_data = np.array(
            [[float(word) for word in line] for line in data[1:]],
            dtype=float,
        )
    return numeric_data, header


def read_340_table(filename: str | Path) -> tuple[NDArray[float], NDArray[float]]:
    data = np.loadtxt(filename, skiprows=9, usecols=(1, 2), unpack=True)
    print(data.dtype, type(data))
    return data[0], data[1]
