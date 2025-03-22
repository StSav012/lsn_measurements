import sys

from contextlib import suppress

from ipaddress import IPv4Address, ip_address
from multiprocessing import Process
from queue import Empty
from multiprocessing.queues import Queue as QueueType

from socket import AF_INET, SOCK_DGRAM, socket
from typing import Any, Callable

if not hasattr(QueueType, "__class_getitem__"):
    # Python < 3.12 or so
    QueueType.__class_getitem__ = lambda *_, **__: QueueType

__all__ = ["Auto", "warning", "error", "get_local_ip", "silent_alive", "drain_queue", "clear_queue_after_process"]

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


def drain_queue(queue: QueueType[Any]) -> None:
    while not queue.empty():
        with suppress(Empty):
            queue.get_nowait()


def clear_queue_after_process(process: Process, *queue: QueueType[Any]) -> None:
    while True:
        while silent_alive(process):
            for q in queue:
                if isinstance(q, QueueType):  # ensure the correct type
                    drain_queue(q)
        for q in queue:
            if isinstance(q, QueueType):
                drain_queue(q)
        if process is None:
            break
        try:
            process.join(timeout=0.1)
            process.close()
        except (
            ValueError,  # ValueError: the process object is closed
            AssertionError,  # AssertionError: can only join a started process
        ):
            break
        except TimeoutError:
            continue
        else:
            break
