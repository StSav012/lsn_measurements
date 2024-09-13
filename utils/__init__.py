# -*- coding: utf-8 -*-
import sys

__all__ = ["Auto", "warning", "error", "get_local_ip"]

from ipaddress import IPv4Address, ip_address

from socket import AF_INET, SOCK_DGRAM, socket

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
