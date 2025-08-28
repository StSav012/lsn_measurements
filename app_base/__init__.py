from abc import ABCMeta

from qtpy.QtWidgets import QWidget

__all__ = ["QWidgetMeta"]


class QWidgetMeta(type(QWidget), ABCMeta):
    pass
