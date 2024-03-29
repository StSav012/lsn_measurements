# coding: utf-8
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from ipaddress import IPv4Address, IPv4Network, ip_address, ip_network
from socket import AF_INET, SOCK_DGRAM, SOCK_STREAM, error, gethostbyname_ex, gethostname, socket

from netifaces import ifaddresses, interfaces

__all__ = ["port_scanner"]


def port_scanner(port: int, timeout: float = 0.1) -> list[IPv4Address]:
    def connectable_host(host_to_try: IPv4Address) -> bool:
        sock: socket = socket(AF_INET, SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((str(host_to_try), port))
        except error:
            return False
        else:
            return True
        finally:
            sock.close()

    def get_local_ip() -> IPv4Address:
        # https://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib/25850698#25850698
        sock: socket = socket(AF_INET, SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))  # connect() for UDP doesn't send packets
        ip: str = sock.getsockname()[0]
        sock.close()
        return ip_address(ip)

    local_ips: set[IPv4Address] = set(map(ip_address, gethostbyname_ex(gethostname())[2]))
    local_ips.add(get_local_ip())

    connectable_hosts: list[IPv4Address] = []

    interface: str
    for interface in interfaces():
        address: dict[str, str]
        for address in ifaddresses(interface).get(AF_INET, []):
            for local_ip in local_ips:
                if address["addr"] != str(local_ip):
                    continue
                network: IPv4Network = ip_network(f'{local_ip}/{address["net""mask"]}', strict=False)
                with ThreadPoolExecutor(max_workers=255) as executor:
                    futures: dict[Future[bool], IPv4Address] = {
                        executor.submit(connectable_host, host): host for host in network.hosts() if host != local_ip
                    }
                    future: Future[bool]
                    for future in as_completed(futures):
                        try:
                            if future.result():
                                connectable_hosts.append(futures[future])
                        except Exception as ex:
                            print(f"{futures[future]} generated an exception: {ex}")

    return connectable_hosts


if __name__ == "__main__":
    print(port_scanner(5025))
