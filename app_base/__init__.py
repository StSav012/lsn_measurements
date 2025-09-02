from abc import ABCMeta
from multiprocessing.sharedctypes import Synchronized

from qtpy.QtWidgets import QWidget

__all__ = ["QWidgetMeta"]


if not hasattr(Synchronized, "__class_getitem__"):
    Synchronized.__class_getitem__ = lambda *_, **__: Synchronized


class QWidgetMeta(type(QWidget), ABCMeta):
    pass
