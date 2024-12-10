# coding=utf-8

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timedelta
from ipaddress import IPv4Address
from sys import maxsize

from telnetlib3 import TelnetReader, TelnetWriter, open_connection

from utils import get_local_ip
from utils.port_scanner import port_scanner

NOOP_TIMEOUT_AFTER_CONNECTION_ESTABLISHED: timedelta = timedelta(seconds=0.2)


def silent_float(string: str) -> float | None:
    try:
        return float(string)
    except ValueError:
        return None


def silent_bool(s: str) -> bool | None:
    if s in ("1", "on") or silent_float(s) not in (0.0, None):
        return True
    elif s in ("0", "off") or silent_float(s) == 0.0:
        return False
    else:
        return None


def dac_v(v: float):
    return max(
        0x00_00_00,
        min(
            0xFF_FF_FF,
            int((v + 10.0) / 20.0 * 0xFF_FF_FF),
        ),
    )


def constrain(x: float, limits: tuple[float, float]) -> float:
    return max(min(limits), min(max(limits), x))


def is_within(x: float, limits: tuple[float, float]) -> bool:
    return min(limits) <= x <= max(limits)


logger: logging.Logger = logging.getLogger("LN/HR DAC II")

LIMITS: dict[int, tuple[float, float]] = {
    1: (-2, 2),
    # 2: (-10, 10),
    3: (-2, 2),
    4: (-2, 2),
    5: (-2, 2),
    6: (-2, 2),
    7: (-2, 2),
    8: (-2, 2),
    9: (-2, 2),
    10: (-2, 2),
    11: (-2, 2),
    12: (-2, 2),
}


async def main(client: str, server: tuple[str, int]) -> None:
    arbitrary_commands_allowed: bool = False
    respect_limits: bool = True

    async def handle_client(server_reader, server_writer) -> None:
        nonlocal arbitrary_commands_allowed, respect_limits

        connection_established: datetime = datetime.now()

        with suppress(ConnectionResetError):
            data: str
            while data := (await server_reader.read(1024)).decode():
                responses: list[str] = []
                response: str

                for line in data.splitlines()[::-1]:
                    if not line:
                        continue
                    for command in line.split(";"):
                        command = command.strip()
                        if not command:
                            logger.warning(f"empty command in {data!r}")
                            continue
                        words: list[str] = command.split()
                        if words[0].startswith(":"):
                            words[0] = words[0][1:]
                        if not words[0]:
                            logger.warning(f"malformed command: {command!r}")
                            continue
                        # noinspection PyTypeChecker
                        words = list(map(str.casefold, words))
                        match words:
                            case ("*idn?",):
                                responses.append("Low Noise/High Resolution DAC II")
                            case ("*rst",):
                                telnet_writer.write("ALL OFF\r\n")
                                await telnet_reader.read(maxsize)
                                telnet_writer.write("ALL 7FFFFF\r\n")
                                await telnet_reader.read(maxsize)
                            case ("power", s):
                                if datetime.now() - connection_established < NOOP_TIMEOUT_AFTER_CONNECTION_ESTABLISHED:
                                    logger.info(
                                        f"skipping {command!r} issued "
                                        f"within {NOOP_TIMEOUT_AFTER_CONNECTION_ESTABLISHED} "
                                        "after the connection had been established"
                                    )
                                    continue
                                if s in ("1", "on") or silent_float(s) not in (0.0, None):
                                    dac_s = "ON"
                                elif s in ("0", "off") or silent_float(s) == 0.0:
                                    dac_s = "OFF"
                                else:
                                    continue
                                telnet_writer.write(f"ALL {dac_s}\r\n")
                                response = await telnet_reader.read(maxsize)
                                logger.debug(f"{command!r}\t->\t{response!r}")
                            case "power?":
                                telnet_writer.write("ALL S?\r\n")
                                response = await telnet_reader.read(maxsize)
                                response = str(int(any(r == "ON\r\n" for r in response.split(";"))))
                                logger.debug(f"{command!r}\t->\t{response!r}")
                            case ("output", ch0, *chs, s):
                                if datetime.now() - connection_established < NOOP_TIMEOUT_AFTER_CONNECTION_ESTABLISHED:
                                    logger.info(
                                        f"skipping {command!r} issued "
                                        f"within {NOOP_TIMEOUT_AFTER_CONNECTION_ESTABLISHED} "
                                        "after the connection had been established"
                                    )
                                    continue
                                match silent_bool(s):
                                    case True:
                                        dac_s = "ON"
                                    case False:
                                        dac_s = "OFF"
                                    case None:
                                        continue
                                telnet_writer.write(";".join((f"{ch} {dac_s}" for ch in (ch0, *chs))) + "\r\n")
                                response = await telnet_reader.read(maxsize)
                                logger.debug(f"{command!r}\t->\t{response!r}")
                            case ("output", ch):
                                if ch.endswith("?"):
                                    telnet_writer.write(f"{ch[:-1]} s?\r\n")
                                    response = await telnet_reader.read(maxsize)
                                    logger.debug(f"{command!r}\t->\t{response!r}")
                                    response = {"OFF\r\n": "0", "ON\r\n": "1"}[response]
                                    responses.append(response)
                            case ("voltage", ch0, *chs, v):
                                if datetime.now() - connection_established < NOOP_TIMEOUT_AFTER_CONNECTION_ESTABLISHED:
                                    continue
                                cmd = ";".join(
                                    (
                                        f"{ch} {dac_v(float(v)):X}"
                                        for ch in (ch0, *chs)
                                        if (
                                            not (respect_limits and int(ch) in LIMITS)
                                            or is_within(float(v), LIMITS[int(ch)])
                                        )
                                    )
                                )
                                if cmd:
                                    telnet_writer.write(cmd + "\r\n")
                                    response = await telnet_reader.read(maxsize)
                                    logger.debug(f"{command!r}\t->\t{response!r}")
                                else:
                                    logger.warning(f"{command!r} not within the set limits")
                            case ("voltage", ch):
                                if ch.endswith("?"):
                                    telnet_writer.write(f"{ch[:-1]} v?\r\n")
                                    response = await telnet_reader.read(maxsize)
                                    logger.debug(f"{command!r}\t->\t{response!r}")
                                    response = str((int(response, 0x10) - 0x7F_FF_FF) / 0x7F_FF_FF * 10.0)
                                    responses.append(response)
                            case "arbitrary_commands_allowed?":
                                responses.append(str(int(arbitrary_commands_allowed)))
                            case ("arbitrary_commands_allowed", s):
                                match silent_bool(s):
                                    case None:
                                        continue
                                    case _ as _b:
                                        b = _b
                                arbitrary_commands_allowed = bool(int(b))
                                logger.warning(
                                    "dis" * (not arbitrary_commands_allowed) + "allowed to execute arbitrary commands"
                                )
                            case "respect_limits?":
                                responses.append(str(int(respect_limits)))
                            case ("respect_limits", s):
                                match silent_bool(s):
                                    case None:
                                        continue
                                    case _ as _b:
                                        b = _b
                                respect_limits = bool(int(b))
                                logger.warning("dis" * (not respect_limits) + "respecting voltage limits")
                            case _:
                                if arbitrary_commands_allowed:
                                    telnet_writer.write(command + "\r\n")
                                    response = await telnet_reader.read(maxsize)
                                    logger.debug(f"{command!r}\t->\t{response!r}")
                                    responses.append(response.rstrip())
                                else:
                                    logger.warning(f"disallowed to execute {command!r}")
                if responses:
                    server_writer.write(";".join(responses).encode() + b"\n")
                    await server_writer.drain()
            server_writer.close()

    telnet_reader: TelnetReader
    telnet_writer: TelnetWriter
    telnet_reader, telnet_writer = await open_connection(client, term="\r\n")
    async with await asyncio.start_server(handle_client, *server) as server:
        await server.serve_forever()


if __name__ == "__main__":
    hosts: list[IPv4Address] = port_scanner(23, 80, 2343, 3079, 3580)
    if not hosts:
        import tkinter.messagebox

        tkinter.messagebox.showerror("Device Not Found", "Could not find the device within the network.")
        exit(1)

    host: str = str(hosts[0])
    port: int = 58110
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s\t%(levelname)s:%(name)s:%(message)s")

    logger.info(f"use TCPIP::{get_local_ip()}::{port}::SOCKET")
    logger.info(
        """commands:
    *idn?                              identification string
    *rst                               disable all channels and set the voltages to 0
    power?                             get whether any channel is on {0 | 1}
    power {0|1}                        set state for all channels {0 | 1 | OFF | ON}
    output <ch>?                       get channel <ch> state {0 | 1}
    output <ch>[ <ch>[...]] {0|1}      set channel(s) <ch> state {0 | 1 | OFF | ON}
    voltage <ch>?                      get voltage for channel <ch> [V]
    voltage <ch>[ <ch>[...]] <value>   set voltage <value> for channel(s) <ch> [V]
    arbitrary_commands_allowed?        check whether commands not listed above are passed directly to the device {0 | 1}
    arbitrary_commands_allowed <value> allow or disallow passing commands directly to the device {0 | 1 | OFF | ON}
    respect_limits?                    check whether voltage limits for channels are respected {0 | 1}
    respect_limits <value>             respect or disrespect voltage limits for channels {0 | 1 | OFF | ON}
"""
    )

    loop = asyncio.get_event_loop()
    try:
        (loop.create_task if loop.is_running() else asyncio.run)(main(host, ("0.0.0.0", port)))
    except KeyboardInterrupt:
        logger.info("interrupted")
