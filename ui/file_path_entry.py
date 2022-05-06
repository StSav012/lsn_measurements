# coding: utf-8

from pathlib import Path
from typing import Optional

import pathvalidate
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QStyle, QWidget

__all__ = ['FilePathEntry']


class FilePathEntry(QWidget):
    path_changed: pyqtSignal = pyqtSignal(str, name='path_changed')

    def __init__(self, initial_file_path: str = '', parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.path: Optional[Path] = None

        layout: QHBoxLayout = QHBoxLayout(self)

        self.text: QLineEdit = QLineEdit(self)
        self.text.setText(initial_file_path)
        self.text.textChanged.connect(self.on_text_changed)
        layout.addWidget(self.text)

        self.status: QLabel = QLabel(self)
        layout.addWidget(self.status)

        self.browse_button: QPushButton = QPushButton('Browse…', self)
        self.browse_button.clicked.connect(self.on_browse_button_clicked)
        layout.addWidget(self.browse_button)

        layout.setStretch(1, 0)
        layout.setStretch(2, 0)
        layout.setContentsMargins(0, 0, 0, 0)

        if initial_file_path:
            self.on_text_changed(initial_file_path, emit=False)

    def on_text_changed(self, text: str, emit: bool = True) -> None:
        """ display an icon showing whether the entered file name is acceptable """

        self.text.setToolTip(text)

        if not text:
            self.status.clear()
            self.path = None
            if emit:
                self.path_changed.emit('')
            return

        path: Path = Path(text).resolve()
        if path.is_dir():
            self.status.setPixmap(self.style().standardIcon(QStyle.SP_MessageBoxCritical).pixmap(self.text.height()))
            self.path = None
            if emit:
                self.path_changed.emit('')
            return
        if not path.exists():
            self.status.setPixmap(self.style().standardIcon(QStyle.SP_MessageBoxCritical).pixmap(self.text.height()))
            self.path = None
            if emit:
                self.path_changed.emit('')
            return
        try:
            pathvalidate.validate_filepath(path, platform='auto')
        except pathvalidate.error.ValidationError as ex:
            print(ex.description)
            self.status.setPixmap(self.style().standardIcon(QStyle.SP_MessageBoxCritical).pixmap(self.text.height()))
            self.path = None
            if emit:
                self.path_changed.emit('')
        else:
            self.status.clear()
            self.path = path
            if emit:
                self.path_changed.emit(str(path))

    def on_browse_button_clicked(self) -> None:
        new_file_name: str
        new_file_name, _ = QFileDialog.getOpenFileName(
            self, 'Open...',
            str(self.path or ''),
            'Calibration File (*.340)')
        if new_file_name:
            self.text.setText(new_file_name)
