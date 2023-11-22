# -*- coding: utf-8 -*-
import sys

__all__ = ["Auto", "warning", "error"]

Auto = None


def warning(msg: str) -> None:
    sys.stderr.write(f"WARNING: {msg}\n")


def error(msg: str) -> None:
    sys.stderr.write(f"ERROR: {msg}\n")
