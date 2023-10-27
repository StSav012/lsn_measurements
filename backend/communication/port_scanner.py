# coding: utf-8
from __future__ import annotations

import ipaddress
import socket
import threading

import netifaces

__all__ = ["port_scanner"]


def port_scanner(port: int, timeout: float = 0.1) -> list[ipaddress.IPv4Address]:
    connectable_hosts: list[ipaddress.IPv4Address] = []

    def append_to_connectable_hosts(host_to_try: ipaddress.IPv4Address) -> None:
        sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((str(host_to_try), port))
        except socket.error:
            pass
        else:
            connectable_hosts.append(host_to_try)
        finally:
            sock.close()

    local_ip: ipaddress.IPv4Address = ipaddress.ip_address(
        socket.gethostbyname(socket.gethostname())
    )

    interface: str
    for interface in netifaces.interfaces():
        address: dict[str, str]
        for address in netifaces.ifaddresses(interface).get(netifaces.AF_INET, []):
            if address["addr"] == str(local_ip):
                network: ipaddress.IPv4Network = ipaddress.ip_network(
                    f'{local_ip}/{address["net""mask"]}', strict=False
                )
                threads: list[threading.Thread] = [
                    threading.Thread(target=append_to_connectable_hosts, args=(host,))
                    for host in network.hosts()
                    if host != local_ip
                ]
                list(map(threading.Thread.start, threads))
                list(map(threading.Thread.join, threads))
    return connectable_hosts


if __name__ == "__main__":
    print(port_scanner(5025))
