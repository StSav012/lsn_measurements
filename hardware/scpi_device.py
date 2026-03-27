import sys
from collections.abc import Callable, Collection
from math import nan
from socket import AF_INET, SOCK_STREAM, socket
from typing import Any, ClassVar, Final, Self

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
        values: type[T] | Collection[str] | tuple[type[T], str, ...],
        *,
        parameter: T | None = None,
        read: bool = True,
        write: bool = True,
        first_matching: bool = False,
        doc: str = "",
    ) -> property:
        getter: Callable[[Any], None] | None
        if not read:
            getter = None
        else:

            def getter(self: Self) -> T:
                if self.socket is None:
                    if values is float:
                        return nan
                    return values()
                if values is bool:
                    return to_bool(self.query(cmd, parameter=parameter))
                if values is int:
                    return int(float(self.query(cmd, parameter=parameter)))
                if (
                    isinstance(values, tuple)
                    and (isinstance(values[0], type) or callable(values[0]))
                    and all(isinstance(i, str) for i in values[1:])
                ):
                    ret: str = self.query(cmd, parameter=parameter)
                    try:
                        return find_single_matching_string(ret, values[1:], first_matching=first_matching)
                    except ValueError:
                        return values[0](ret)
                if isinstance(values, Collection):
                    if not all(isinstance(i, str) for i in values):
                        raise TypeError(
                            f"Expected a collection of strings, got {type(values)}({[type(i) for i in values]})"
                        )
                    return find_single_matching_string(
                        self.query(cmd, parameter=parameter), values, first_matching=first_matching
                    )
                return values(self.query(cmd, parameter=parameter))

        setter: Callable[[Any, Any], None] | None
        if not write:
            setter = None
        else:

            def setter(self: Self, new_value: T) -> None:
                if self.socket is None:
                    return
                if isinstance(values, tuple):
                    if isinstance(new_value, str):
                        try:
                            self.issue(
                                cmd,
                                find_single_matching_string(new_value, values, first_matching=first_matching),
                            )
                        except ValueError:
                            self.issue(cmd, values[0](new_value))
                    else:
                        self.issue(cmd, values[0](new_value))
                elif isinstance(values, Collection):
                    if not (isinstance(new_value, str) and all(isinstance(i, str) for i in values)):
                        raise TypeError("Incompatible types", type(new_value), [type(i) for i in values])
                    self.issue(cmd, find_single_matching_string(new_value, values, first_matching=first_matching))
                else:
                    self.issue(cmd, values(new_value))

        return property(getter, setter, None, doc or (f"Query and set {cmd}" if setter is not None else f"Query {cmd}"))

    def __del__(self) -> None:
        if self.socket is not None:
            self.socket.close()

    idn: property = property_by_command("*idn?", str, write=False)
    opc: property = property_by_command("*opc?", bool, write=False)

    def reset(self) -> None:
        self.communicate("*rst")

    def communicate(self, command: str) -> str | None:
        if self.socket is None:
            return ""
        self.socket.send((command.strip()).encode() + self.terminator)
        if not command.split()[0].endswith("?"):
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
    prefix: ClassVar[str] = ""
    sub_prefix: ClassVar[str] = ""

    def __init__(self, parent: SCPIDevice, prefix: str = prefix) -> None:
        if not prefix:
            raise ValueError("Specify the sub-category prefix")

        self.parent: Final[SCPIDevice] = parent
        if self.__class__.sub_prefix:
            if (
                ":" not in self.__class__.sub_prefix.lstrip(":")
                and self.__class__.sub_prefix.casefold() != self.__class__.__name__.strip("_").casefold()
            ):
                print(
                    f"WARNING: Class name {self.__class__.__name__} doesn't match the sub-category prefix {self.__class__.sub_prefix}",
                    file=sys.stderr,
                )
            prefix = ":".join((prefix, self.__class__.sub_prefix.lstrip(":")))
        self._prefix: Final[str] = prefix

    def _make_command(self, cmd: str) -> str:
        cmd = cmd.lstrip(":")
        return ":".join((self._prefix, cmd)) if cmd else self._prefix

    @staticmethod
    def subproperty_by_command[T](
        cmd: str,
        values: type[T] | Collection[str] | tuple[type[T], str, ...],
        *,
        parameter: T | None = None,
        read: bool = True,
        write: bool = True,
        first_matching: bool = False,
        doc: str = "",
    ) -> property:
        getter: Callable[[Any], None] | None
        if not read:
            getter = None
        else:

            def getter(self: SCPIDeviceSubCategory) -> T:
                if self.parent.socket is None:
                    if values is float:
                        return nan
                    return values()
                subcmd: str = self._make_command(cmd)
                if values is bool:
                    return to_bool(self.parent.query(subcmd, parameter=parameter))
                if values is int:
                    return int(float(self.parent.query(subcmd, parameter=parameter)))
                if (
                    isinstance(values, tuple)
                    and (isinstance(values[0], type) or callable(values[0]))
                    and all(isinstance(i, str) for i in values[1:])
                ):
                    ret: str = self.parent.query(subcmd, parameter=parameter)
                    try:
                        return find_single_matching_string(ret, values[1:], first_matching=first_matching)
                    except ValueError:
                        return values[0](ret)
                elif isinstance(values, Collection):
                    if not all(isinstance(i, str) for i in values):
                        raise TypeError(
                            f"Expected a collection of strings, got {type(values)}({[type(i) for i in values]})"
                        )
                    return find_single_matching_string(
                        self.parent.query(subcmd, parameter=parameter), values, first_matching=first_matching
                    )
                return values(self.parent.query(subcmd, parameter=parameter))

        setter: Callable[[Any, Any], None] | None
        if not write:
            setter = None
        else:

            def setter(self: SCPIDeviceSubCategory, new_value: T) -> None:
                if self.parent.socket is None:
                    return
                subcmd: str = self._make_command(cmd)
                if isinstance(values, tuple):
                    if isinstance(new_value, str):
                        try:
                            self.parent.issue(
                                subcmd,
                                find_single_matching_string(new_value, values, first_matching=first_matching),
                            )
                        except ValueError:
                            if callable(values[0]):
                                self.parent.issue(subcmd, values[0](new_value))
                            else:
                                raise
                    else:
                        self.parent.issue(subcmd, values[0](new_value))
                elif isinstance(values, Collection):
                    if not (isinstance(new_value, str) and all(isinstance(i, str) for i in values)):
                        raise TypeError("Incompatible types", type(new_value), [type(i) for i in values])
                    self.parent.issue(
                        subcmd, find_single_matching_string(new_value, values, first_matching=first_matching)
                    )
                else:
                    self.parent.issue(subcmd, values(new_value))

        return property(getter, setter, None, doc or (f"Query and set {cmd}" if setter is not None else f"Query {cmd}"))

    def communicate(self, command: str) -> str | None:
        self.parent.communicate(self._make_command(command))

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
