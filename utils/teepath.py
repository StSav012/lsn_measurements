import sys
from io import BytesIO, TextIOWrapper
from pathlib import WindowsPath
from typing import Literal, TextIO

__all__ = ["TeeBytesIO", "TeePath", "TeeTextIOWrapper"]


class TeeBytesIO(BytesIO):
    def write(self, /, __s: bytes) -> int:
        sys.stdout.write(repr(__s))
        return super().write(__s)


class TeeTextIOWrapper(TextIOWrapper):
    def write(self, /, __s: str) -> int:
        sys.stdout.write(__s)
        return super().buffer.write(__s)


OpenTextModeWriting = Literal["w", "wt", "tw", "a", "at", "ta", "x", "xt", "tx"]


class TeePath(WindowsPath):
    def open(
        self,
        mode: OpenTextModeWriting = "wt",
        buffering: int = 1,
        encoding: str | None = "utf-8",
        errors: str | None = None,
        newline: str | None = None,
    ) -> TeeTextIOWrapper:
        return TeeTextIOWrapper(super().open(mode, buffering, encoding, errors, newline))

    def append_text(self, data: str) -> int:
        f_out: TextIO
        with self.open("at") as f_out:
            return f_out.write(data)
