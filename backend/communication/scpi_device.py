# coding: utf-8
import socket
from socket import *
from typing import Any, List, Optional

from backend.communication.port_scanner import port_scanner
from backend.utils import warning

__all__ = ['SCPIDevice']


class SCPIDevice:
    def __init__(self, ip: Optional[str], port: int, *, terminator: bytes = b'\r\n', expected: bool = True,
                 reset: bool = True) -> None:

        if ip is None and expected:
            from ipaddress import IPv4Address

            connectable_hosts: List[IPv4Address] = port_scanner(port)
            if not connectable_hosts:
                raise RuntimeError(f'{self.__class__.__name__} with open port {port} could not be found automatically. '
                                   'Try specifying an IP address.')
            if len(connectable_hosts) > 1:
                raise RuntimeError(f'There are numerous devices with open port {port}:\n',
                                   ',\n'.join(map(str, connectable_hosts)),
                                   '\nTry specifying an IP address.')
            ip = str(connectable_hosts[0])

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
                warning(f'{self.__class__.__name__} not connected.')
            else:
                if reset:
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
