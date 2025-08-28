import sys
from collections.abc import Callable
from select import select
from sys import stdin
from typing import TypeVar

__all__ = [
    "ask_float",
    "ask_int",
    "ask_ok_cancel",
    "ask_question",
    "ask_retry_cancel",
    "ask_string",
    "ask_yes_no",
    "ask_yes_no_cancel",
    "show_error",
    "show_info",
    "show_warning",
]

_T = TypeVar("_T")
try:
    from typing import ParamSpec
except ImportError:
    ParamSpec = str
_P = ParamSpec("_P")


def _show[P, T](method: Callable[P, T], title: str, message: str) -> T | str | None:
    from threading import Semaphore, Thread

    s: Semaphore = Semaphore(2)

    res: list[T | str] = []

    def show_gui_message() -> None:
        _kwargs: dict[str, str]
        if "\n\n" in message:
            _kwargs = dict(zip(["message", "detail"], message.split("\n\n", maxsplit=1), strict=False))
        else:
            _kwargs = {"message": message}
        with s:
            res.append(method(title=title, **_kwargs))

    def show_cli_message() -> None:
        with s:
            sys.stdout.write(f"{title}: {message}" if title else message)
            select([stdin], [], [])
            res.append(input())

    a: Thread = Thread(target=show_gui_message, daemon=True)
    b: Thread = Thread(target=show_cli_message, daemon=True)
    a.start()
    b.start()

    with s:
        a.join(0)
        b.join(0)

    if res:
        return res[0]
    return None


def show_info(title: str, message: str) -> str | None:
    from tkinter.messagebox import showinfo

    return _show(method=showinfo, title=title, message=message)


def show_warning(title: str, message: str) -> str | None:
    from tkinter.messagebox import showwarning

    return _show(method=showwarning, title=title, message=message)


def show_error(title: str, message: str) -> str | None:
    from tkinter.messagebox import showerror

    return _show(method=showerror, title=title, message=message)


def ask_question(title: str, message: str) -> str | None:
    from tkinter.messagebox import askquestion

    return _show(method=askquestion, title=title, message=message)


def ask_yes_no(title: str, message: str) -> bool | str | None:
    from tkinter.messagebox import askyesno

    return _show(method=askyesno, title=title, message=message)


def ask_ok_cancel(title: str, message: str) -> bool | str | None:
    from tkinter.messagebox import askokcancel

    return _show(method=askokcancel, title=title, message=message)


def ask_yes_no_cancel(title: str, message: str) -> bool | str | None:
    from tkinter.messagebox import askyesnocancel

    return _show(method=askyesnocancel, title=title, message=message)


def ask_retry_cancel(title: str, message: str) -> bool | str | None:
    from tkinter.messagebox import askretrycancel

    return _show(method=askretrycancel, title=title, message=message)


def _ask[P, T](method: Callable[P, T], title: str, prompt: str) -> T | str | None:
    from threading import Semaphore, Thread

    s: Semaphore = Semaphore(2)

    res: list[T | str] = []

    def show_gui_message() -> None:
        with s:
            res.append(method(title=title, prompt=prompt))

    def show_cli_message() -> None:
        with s:
            sys.stdout.write(f"{title}: {prompt}" if title else prompt)
            select([stdin], [], [])
            res.append(input())

    a: Thread = Thread(target=show_gui_message, daemon=True)
    b: Thread = Thread(target=show_cli_message, daemon=True)
    a.start()
    b.start()

    with s:
        a.join(0)
        b.join(0)

    if res:
        return res[0]
    return None


def ask_string(title: str, prompt: str) -> str | None:
    from tkinter.simpledialog import askstring

    return _ask(method=askstring, title=title, prompt=prompt)


def ask_int(title: str, prompt: str) -> int | str | None:
    from tkinter.simpledialog import askinteger

    return _ask(method=askinteger, title=title, prompt=prompt)


def ask_float(title: str, prompt: str) -> float | str | None:
    from tkinter.simpledialog import askfloat

    return _ask(method=askfloat, title=title, prompt=prompt)
