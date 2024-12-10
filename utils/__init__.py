# -*- coding: utf-8 -*-
import sys

from contextlib import suppress

from ipaddress import IPv4Address, ip_address
from multiprocessing import Process
from queue import Empty, Queue

from socket import AF_INET, SOCK_DGRAM, socket
from typing import Any, Callable

__all__ = ["Auto", "warning", "error", "get_local_ip", "silent_alive", "clear_queue_after_process"]

Auto = None


def warning(msg: str) -> None:
    sys.stderr.write(f"WARNING: {msg}\n")


def error(msg: str) -> None:
    sys.stderr.write(f"ERROR: {msg}\n")


def get_local_ip() -> IPv4Address:
    # https://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib/25850698#25850698
    sock: socket = socket(AF_INET, SOCK_DGRAM)
    sock.connect(("8.8.8.8", 80))  # connect() for UDP doesn't send packets
    ip: str = sock.getsockname()[0]
    sock.close()
    return ip_address(ip)


def silent_alive(process: Any) -> bool:
    is_alive: bool | Callable[[], bool] = getattr(process, "is_alive", False)
    if callable(is_alive):
        with suppress(ValueError):
            return process.is_alive()
    return bool(is_alive)


def clear_queue_after_process(process: Process, queue: Queue) -> None:
    while silent_alive(process):
        while queue.qsize():
            with suppress(Empty):
                queue.get_nowait()
    while queue.qsize():
        with suppress(Empty):
            queue.get_nowait()
    with suppress(
        ValueError,  # ValueError: the process object is closed
        AssertionError,  # AssertionError: can only join a started process
        AttributeError,  # AttributeError: 'NoneType' object has no attribute 'join'
    ):
        process.join()
        process.close()
