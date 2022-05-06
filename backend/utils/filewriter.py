# -*- coding: utf-8 -*-
import time
from pathlib import Path
from threading import Lock, Thread
from typing import Iterable, List, NamedTuple, TextIO, Union

import numpy as np
from numpy.typing import NDArray

__all__ = ['FileWriter']

QueueRecord = NamedTuple('QueueRecord', file_name=Path, file_mode=str, x=Union[NDArray[np.float64], Iterable[float]])


class FileWriter(Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.daemon = True

        self.queue: List[QueueRecord] = []
        self.lock: Lock = Lock()
        self.done: bool = False

    def __del__(self) -> None:
        self._write_queue()
        self.done = True

    def write(self, file_name: Path, file_mode: str, x: Union[NDArray[np.float64], Iterable[float]]) -> None:
        with self.lock:
            self.queue.append(QueueRecord(file_name, file_mode, x))

    def _write_queue(self) -> None:
        if not self.done and not self.queue:
            time.sleep(0.05)
        while self.queue:
            with self.lock:
                qr: QueueRecord = self.queue.pop(0)
            f_out: TextIO
            with qr.file_name.open(qr.file_mode) as f_out:
                if isinstance(qr.x, np.ndarray) and qr.x.ndim == 2:
                    for x in qr.x.T:
                        f_out.write('\t'.join(f'{y}' for y in x) + '\n')
                else:
                    f_out.write('\t'.join(f'{x}' for x in qr.x) + '\n')

    def run(self) -> None:
        while not self.done:
            self._write_queue()
