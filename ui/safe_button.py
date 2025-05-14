import math

from qtpy.QtCore import QTimer, Signal
from qtpy.QtWidgets import QGroupBox, QPushButton, QVBoxLayout, QWidget

__all__ = ["SafeButton"]


class SafeButton(QGroupBox):
    clicked: Signal = Signal(name="clicked")

    def __init__(
        self,
        text: str,
        timeout: float = math.inf,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.timeout: float = timeout
        self._timer: QTimer = QTimer(self)
        self._timer.timeout.connect(self._on_timeout)

        layout: QVBoxLayout = QVBoxLayout(self)

        self._button: QPushButton = QPushButton(text, self)
        self._button.setCheckable(True)
        self._button.toggled.connect(self._on_button_toggled)
        layout.addWidget(self._button)

        self.setCheckable(True)
        self.setChecked(False)
        self.toggled.connect(self._on_toggled)

    def _on_button_toggled(self, on: bool) -> None:
        self.setChecked(False)
        if on:
            self.clicked.emit()

    def _on_toggled(self, on: bool) -> None:
        if on and not math.isinf(self.timeout):
            self._timer.start(round(self.timeout * 1000))
        else:
            self._timer.stop()

    def _on_timeout(self) -> None:
        self.blockSignals(True)
        self.setChecked(False)
        self.blockSignals(False)

    def setText(self, text: str) -> None:
        self._button.setText(text)

    def reset(self) -> None:
        self.blockSignals(True)
        self.setChecked(False)
        self._button.setChecked(False)
        self.blockSignals(False)

    def isPushed(self) -> bool:
        return self._button.isChecked()
