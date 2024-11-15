# coding: utf-8
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from ipaddress import IPv4Address, IPv4Network, ip_address, ip_network
from socket import AF_INET, SOCK_STREAM, error, gethostbyname_ex, gethostname, socket

from netifaces import ifaddresses, interfaces

__all__ = ["port_scanner"]

from utils import get_local_ip


def port_scanner(*ports: int, timeout: float = 0.1) -> list[IPv4Address]:
    ports = list(ports)

    if not ports:
        return []

    def connectable_host(host_to_try: IPv4Address, port_to_try: int) -> bool:
        sock: socket = socket(AF_INET, SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((str(host_to_try), port_to_try))
        except error:
            return False
        else:
            return True
        finally:
            sock.close()

    local_ips: set[IPv4Address] = set(map(ip_address, gethostbyname_ex(gethostname())[2]))
    local_ips.add(get_local_ip())

    connectable_hosts: dict[int, set[IPv4Address]] = defaultdict(set)

    interface: str
    for interface in interfaces():
        address: dict[str, str]
        for address in ifaddresses(interface).get(AF_INET, []):
            for local_ip in local_ips:
                if address["addr"] != str(local_ip):
                    continue
                network: IPv4Network = ip_network(f'{local_ip}/{address["net""mask"]}', strict=False)
                with ThreadPoolExecutor(max_workers=255) as executor:
                    host: IPv4Address
                    port: int
                    futures: dict[Future[bool], (IPv4Address, int)] = {
                        executor.submit(connectable_host, host, port): (host, port)
                        for port in ports
                        for host in network.hosts()
                        if host != local_ip
                    }
                    future: Future[bool]
                    for future in as_completed(futures):
                        try:
                            if future.result():
                                host, port = futures[future]
                                connectable_hosts[port].add(host)
                        except Exception as ex:
                            print(f"{futures[future]} generated an exception: {ex}")

    if not connectable_hosts:
        return []

    return sorted(connectable_hosts[ports.pop()].intersection(*[connectable_hosts[port] for port in ports]))


if __name__ == "__main__":
    print(port_scanner(5025))
