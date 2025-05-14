import sys
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from ipaddress import IPv4Address, IPv4Network, ip_network
from socket import AF_INET, SOCK_STREAM, socket

from netifaces import ifaddresses, interfaces

__all__ = ["port_scanner"]


def port_scanner(
    *ports: int,
    timeout: float = 0.1,
    max_network_capacity: int = 0x100,
) -> list[IPv4Address]:
    ports = list(ports)

    if not ports:
        return []

    def connectable_host(host_to_try: IPv4Address, port_to_try: int) -> bool:
        sock: socket = socket(AF_INET, SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((str(host_to_try), port_to_try))
        except OSError:
            return False
        else:
            return True
        finally:
            sock.close()

    connectable_hosts: dict[int, set[IPv4Address]] = defaultdict(set)

    interface: str
    for interface in interfaces():
        address: dict[str, str]
        for address in ifaddresses(interface).get(AF_INET, []):
            network: IPv4Network = ip_network(f"{address['addr']}/{address['netmask']}", strict=False)

            if network.num_addresses > max_network_capacity:
                continue

            with ThreadPoolExecutor(max_workers=255) as executor:
                host: IPv4Address
                port: int
                futures: dict[Future[bool], (IPv4Address, int)] = {
                    executor.submit(connectable_host, host, port): (host, port)
                    for port in ports
                    for host in network.hosts()
                    if host != address["addr"]
                }
                future: Future[bool]
                for future in as_completed(futures):
                    try:
                        if future.result():
                            host, port = futures[future]
                            connectable_hosts[port].add(host)
                    except Exception as ex:
                        sys.stderr.write(f"{futures[future]} generated an exception: {ex}\n")

    if not connectable_hosts:
        return []

    return sorted(connectable_hosts[ports.pop()].intersection(*[connectable_hosts[port] for port in ports]))


if __name__ == "__main__":
    print(port_scanner(5025))
