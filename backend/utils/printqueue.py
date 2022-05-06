# -*- coding: utf-8 -*-
import sys
import time
from threading import Lock, Thread
from typing import Any, List, NamedTuple

__all__ = ['PrintQueue']


PrintQueueRecord = NamedTuple('PrintQueueRecord', data=str)


class PrintQueue(Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.daemon = True

        self.queue: List[PrintQueueRecord] = []
        self.lock: Lock = Lock()
        self.done: bool = False

    def __del__(self) -> None:
        self._write_queue()
        self.done = True

    def write(self, *data: Any, sep: str = ' ', end: str = '\n') -> None:
        with self.lock:
            self.queue.append(PrintQueueRecord(sep.join(map(str, data)) + end))

    def _write_queue(self) -> None:
        if not self.done and not self.queue:
            time.sleep(0.05)
        while self.queue:
            with self.lock:
                qr: PrintQueueRecord = self.queue.pop(0)
                sys.stdout.write(qr.data)
                sys.stdout.flush()

    def run(self) -> None:
        while not self.done:
            self._write_queue()
