from collections.abc import Callable, Collection
from math import nan
from socket import AF_INET, SOCK_STREAM, socket
from typing import Any, ClassVar, Final

from utils import warning
from utils.port_scanner import port_scanner
from utils.string_utils import find_single_matching_string, to_bool

__all__ = ["SCPIDevice", "SCPIDeviceSubCategory"]


class SCPIDevice:
    def __init__(
        self,
        ip: str | None,
        port: int,
        *,
        terminator: bytes = b"\r\n",
        expected: bool = True,
        reset: bool = True,
    ) -> None:
        self.socket: socket | None = None
        self.terminator: bytes = terminator

        if ip is None and expected:
            from ipaddress import IPv4Address

            connectable_hosts: list[IPv4Address] = port_scanner(port)
            if not connectable_hosts:
                raise RuntimeError(
                    f"{self.__class__.__name__} with open port {port} could not be found automatically. "
                    "Try specifying an IP address.",
                )
            if len(connectable_hosts) > 1:
                raise RuntimeError(
                    f"There are numerous devices with open port {port}:\n",
                    ",\n".join(map(str, connectable_hosts)),
                    "\nTry specifying an IP address.",
                )
            ip = str(connectable_hosts[0])

        if expected:
            self.socket = socket(AF_INET, SOCK_STREAM)
            try:
                self.socket.settimeout(1)
                self.socket.connect((ip, port))
                self.socket.settimeout(None)
            except (TimeoutError, OSError):
                self.socket.close()
                self.socket = None
                warning(f"{self.__class__.__name__} not connected.")
            else:
                if reset:
                    self.reset()

    @staticmethod
    def property_by_command[T](
        cmd: str,
        values: type[T] | Collection[str],
        *,
        parameter: T | None = None,
        read_only: bool = False,
        doc: str = "",
    ) -> property:
        def getter(self: "SCPIDevice") -> T:
            if self.socket is None:
                if values is float:
                    return nan
                return values()
            if values is bool:
                return to_bool(self.query(cmd, parameter=parameter))
            if values is int:
                return int(float(self.query(cmd, parameter=parameter)))
            if isinstance(values, Collection):
                return find_single_matching_string(self.query(cmd, parameter=parameter), values)
            return values(self.query(cmd, parameter=parameter))

        setter: Callable[[Any, Any], None] | None
        if read_only:
            setter = None
        else:

            def setter(self: "SCPIDevice", new_value: T) -> None:
                if self.socket is None:
                    return
                if isinstance(values, Collection):
                    self.issue(cmd, find_single_matching_string(new_value, values))
                else:
                    self.issue(cmd, values(new_value))

        return property(getter, setter, None, doc or (f"Query and set {cmd}" if setter is not None else f"Query {cmd}"))

    def __del__(self) -> None:
        if self.socket is not None:
            self.socket.close()

    idn: property = property_by_command("*idn?", str, read_only=True)
    opc: property = property_by_command("*opc?", bool, read_only=True)

    def reset(self) -> None:
        self.communicate("*rst")

    def communicate(self, command: str) -> str | None:
        if self.socket is None:
            return ""
        self.socket.send((command.strip()).encode() + self.terminator)
        if not command.endswith("?"):
            return None
        resp: bytes = b""
        while not resp.endswith(self.terminator):
            resp += self.socket.recv(1)
            if not resp:
                return ""
        return resp.decode("ascii").strip()

    def query(self, command: str, parameter: object | None = None) -> str:
        command = command.strip()
        if not command.endswith("?"):
            command += "?"
        if parameter is not None:
            return self.communicate(command + " " + str(parameter))
        return self.communicate(command)

    def issue(self, command: str, value: object | None = None) -> None:
        if value is None:
            self.communicate(command.rstrip("?"))
        if isinstance(value, bool):
            value = {False: "OFF", True: "ON"}[value]
        self.communicate(command.rstrip("?") + " " + str(value).rstrip("?"))


class SCPIDeviceSubCategory:
    prefix: ClassVar[str] = ...

    def __init__(self, parent: SCPIDevice) -> None:
        if self.__class__.prefix is ...:
            raise ValueError("Specify the sub-category prefix")

        self.parent: Final[SCPIDevice] = parent

    @staticmethod
    def subproperty_by_command[T](
        cmd: str,
        values: type[T] | Collection[str],
        *,
        parameter: T | None = None,
        read_only: bool = False,
        doc: str = "",
    ) -> property:
        def getter(self: SCPIDeviceSubCategory) -> T:
            if self.parent.socket is None:
                if values is float:
                    return nan
                return values()
            subcmd: str = ":".join((self.__class__.prefix, cmd)) if cmd else self.__class__.prefix
            if values is bool:
                return to_bool(self.parent.query(subcmd, parameter=parameter))
            if values is int:
                return int(float(self.parent.query(subcmd, parameter=parameter)))
            if isinstance(values, Collection):
                return find_single_matching_string(self.parent.query(subcmd, parameter=parameter), values)
            return values(self.parent.query(subcmd, parameter=parameter))

        setter: Callable[[Any, Any], None] | None
        if read_only:
            setter = None
        else:

            def setter(self: SCPIDeviceSubCategory, new_value: T) -> None:
                if self.parent.socket is None:
                    return
                subcmd: str = ":".join((self.__class__.prefix, cmd)) if cmd else self.__class__.prefix
                if isinstance(values, Collection):
                    self.parent.issue(subcmd, find_single_matching_string(new_value, values))
                else:
                    self.parent.issue(subcmd, values(new_value))

        return property(getter, setter, None, doc or (f"Query and set {cmd}" if setter is not None else f"Query {cmd}"))
