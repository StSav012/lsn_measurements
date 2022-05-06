# coding: utf-8
import socket
from socket import *
from typing import Any, Optional

__all__ = ['SCPIDevice']


class SCPIDevice:
    def __init__(self, ip: str, port: int, *, terminator: bytes = b'\r\n', expected: bool = True) -> None:
        self.socket: Optional[socket] = None
        self.terminator: bytes = terminator
        if expected:
            self.socket = socket(AF_INET, SOCK_STREAM)
            try:
                self.socket.settimeout(1)
                self.socket.connect((ip, port))
                self.socket.settimeout(None)
            except (TimeoutError, OSError):
                self.socket.close()
                self.socket = None
            else:
                self.reset()

    def __del__(self):
        if self.socket is not None:
            self.socket.close()

    @property
    def idn(self) -> str:
        return self.communicate('*idn?')

    def opc(self) -> bool:
        return bool(self.communicate('*opc?'))

    def reset(self) -> None:
        self.communicate('*rst')

    def communicate(self, command: str) -> Optional[str]:
        if self.socket is None:
            return ''
        self.socket.send((command.strip()).encode() + self.terminator)
        if not command.endswith('?'):
            return
        resp: bytes = b''
        while not resp.endswith(self.terminator):
            resp += self.socket.recv(1)
            if not resp:
                return ''
        return resp.decode().strip()

    def query(self, command: str) -> str:
        command = command.strip()
        if not command.endswith('?'):
            command += '?'
        return self.communicate(command)

    def issue(self, command: str, value: Any) -> None:
        if isinstance(value, bool):
            value = {False: 'OFF', True: 'ON'}[value]
        self.communicate(command.rstrip('?') + ' ' + str(value).rstrip('?'))
