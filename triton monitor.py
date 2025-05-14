import sys
from collections.abc import Hashable, Iterable, Iterator
from datetime import UTC, datetime
from typing import ClassVar, SupportsIndex, cast, final

from astropy.units import Quantity
from qtpy.QtCore import QByteArray, QSettings, QThread, Qt, Signal, Slot
from qtpy.QtGui import QAction, QCloseEvent, QIcon
from qtpy.QtWidgets import (
    QApplication,
    QDockWidget,
    QFormLayout,
    QLabel,
    QLayoutItem,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from hardware.triton import TritonScript


class OrderedSet:
    def __init__(self, items: Iterable[Hashable] | None = None) -> None:
        self._items: list[Hashable] = []
        if items is not None:
            self._items = [item for item in items if item not in self._items]

    def __repr__(self) -> str:
        return "{" + ", ".join(repr(item) for item in self._items) + "}"

    def __bool__(self) -> bool:
        return bool(self._items)

    def __sub__(self, other: object) -> "OrderedSet":
        if not isinstance(other, OrderedSet):
            raise TypeError(f"Incompatible type {type(other)}")
        return OrderedSet(item for item in self._items if item not in other._items)

    def __iter__(self) -> Iterator[Hashable]:
        yield from self._items

    def add(self, item: Hashable) -> None:
        if item not in self._items:
            self._items.append(item)

    def pop(self, index: SupportsIndex = -1) -> Hashable:
        return self._items.pop(index)

    def update(self, items: "Iterable[Hashable] | OrderedSet") -> None:
        return self._items.extend(item for item in items if item not in self._items)

    def union(self, items: "Iterable[Hashable] | OrderedSet") -> "OrderedSet":
        new_ordered_set: OrderedSet = OrderedSet(self._items)
        new_ordered_set.update(items)
        return new_ordered_set


def all_fields(fields_set: OrderedSet) -> list[str]:
    class TwoSides:
        def __init__(self, to_the_left: Iterable[str] = (), to_the_right: Iterable[str] = ()) -> None:
            self.to_the_left: OrderedSet = OrderedSet(to_the_left)
            self.to_the_right: OrderedSet = OrderedSet(to_the_right)

        def __repr__(self) -> str:
            return f"{{to the left: {self.to_the_left}, to the right: {self.to_the_right}}}"

    other_fields: dict[str, TwoSides] = {}
    index: int
    field: str
    chunk: str
    for chunk in fields_set:
        for index, field in enumerate(chunk):
            if field not in other_fields:
                other_fields[field] = TwoSides(to_the_left=chunk[:index], to_the_right=chunk[index + 1 :])
            else:
                other_fields[field].to_the_left.update(chunk[:index])
                other_fields[field].to_the_right.update(chunk[index + 1 :])

    already_sorted_fields: OrderedSet = OrderedSet()

    def sort(s: OrderedSet, excluded: OrderedSet) -> list[str]:
        s -= already_sorted_fields
        if not s:
            return []
        f: str = cast("str", s.pop())
        already_sorted_fields.add(f)
        if not s:
            return [f]
        return (
            sort(other_fields[f].to_the_left - excluded, other_fields[f].to_the_right.union(excluded))
            + [f]
            + sort(other_fields[f].to_the_right - excluded, other_fields[f].to_the_left.union(excluded))
        )

    return sort(OrderedSet(other_fields.keys()), OrderedSet())


class Poller(QThread):
    statusReceived: ClassVar[Signal] = Signal(dict)
    pressuresReceived: ClassVar[Signal] = Signal(dict)
    thermometryReceived: ClassVar[Signal] = Signal(list, dict)

    def run(self) -> None:
        ts: TritonScript = TritonScript()

        while self.isRunning() and not self.isInterruptionRequested():
            self.statusReceived.emit(ts.status)
            self.pressuresReceived.emit(ts.pressures)
            self.thermometryReceived.emit(*ts.thermometry)


class DictDisplay(QDockWidget):
    def __init__(self, parent: QWidget | None = None, title: str = "") -> None:
        super().__init__(parent)

        if title:
            self.setWindowTitle(title)
            self.setObjectName(title)

        widget: QWidget = QWidget(self)
        self._layout: QFormLayout = QFormLayout(widget)
        self.setWidget(widget)

    def display(self, data: dict[str, bool | None | Quantity]) -> None:
        for row, (key, value) in enumerate(data.items()):
            if row < self._layout.rowCount():
                label: QLabel = self._layout.itemAt(row, QFormLayout.ItemRole.LabelRole).widget()
                if label.text() != key:
                    label.setText(key)
                field: QLabel = self._layout.itemAt(row, QFormLayout.ItemRole.FieldRole).widget()
                if field.text() != str(value):
                    field.setText(str(value))
            else:
                self._layout.addRow(QLabel(key, self), QLabel(str(value), self))
        for row in range(len(data), self._layout.rowCount()):
            label_item: QLayoutItem = self._layout.itemAt(row, QFormLayout.ItemRole.LabelRole)
            field_item: QLayoutItem = self._layout.itemAt(row, QFormLayout.ItemRole.FieldRole)
            self.layout().removeWidget(label_item)
            self.layout().removeWidget(field_item)


class ListDisplay(QTableWidget):
    def __init__(self, parent: QWidget | None = None, title: str = "") -> None:
        super().__init__(parent)

        if title:
            self.setObjectName(title)

        self.setAlternatingRowColors(True)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setMouseTracking(True)

    def display(self, data: list[dict[str, float | str | bool | datetime]]) -> None:
        header: list[str] = all_fields(OrderedSet([tuple(line) for line in data]))
        self.setRowCount(len(data))
        self.setColumnCount(len(header))
        self.setHorizontalHeaderLabels(header)
        for row, line in enumerate(data):
            for col, key in enumerate(header):
                value: float | str | bool | datetime = line.get(key, "")
                if isinstance(value, datetime):
                    value = value.replace(tzinfo=UTC).astimezone()
                value = str(value)
                item: QTableWidgetItem | None = self.item(row, col)
                if item is None:
                    self.setItem(row, col, QTableWidgetItem(str(value)))
                elif item.text() != value:
                    item.setText(value)


@final
class MainWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.settings: QSettings = QSettings("SavSoft", "Triton Monitor", self)

        self.setWindowTitle(self.tr("Triton Monitor"))
        self.setWindowIcon(QIcon("lsn.svg"))

        self.menu_bar: QMenuBar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)
        self.view_menu: QMenu = self.menu_bar.addMenu(self.tr("&View"))
        self.help_menu: QMenu = self.menu_bar.addMenu(self.tr("&Help"))

        self.status_panel: DictDisplay = DictDisplay(self, self.tr("Status"))
        self.pressures_panel: DictDisplay = DictDisplay(self, self.tr("Pressures"))
        self.heaters_panel: DictDisplay = DictDisplay(self, self.tr("Heaters"))
        self.sensors_table: ListDisplay = ListDisplay(self, self.tr("Sensors"))

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.status_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.pressures_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.heaters_panel)
        self.setCentralWidget(self.sensors_table)

        self.view_menu.addAction(self.status_panel.toggleViewAction())
        self.view_menu.addAction(self.pressures_panel.toggleViewAction())
        self.view_menu.addAction(self.heaters_panel.toggleViewAction())
        self.help_menu.addAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarMenuButton),
            self.tr("About &Qt…"),
            lambda: QMessageBox.aboutQt(self),
        ).setMenuRole(QAction.MenuRole.AboutQtRole)

        self.worker: Poller = Poller(self)
        self.worker.statusReceived.connect(self.on_status_received)
        self.worker.pressuresReceived.connect(self.on_pressures_received)
        self.worker.thermometryReceived.connect(self.on_thermometry_received)
        self.worker.finished.connect(self.close)
        self.worker.start()

        self.load_settings()

    def load_settings(self) -> None:
        self.settings.beginGroup("geometry")
        self.restoreGeometry(cast("QByteArray", self.settings.value("window", QByteArray())))
        self.sensors_table.restoreGeometry(cast("QByteArray", self.settings.value("sensors", QByteArray())))
        self.sensors_table.horizontalHeader().restoreGeometry(
            self.settings.value("sensorsHorizontalHeader", QByteArray()),
        )
        self.sensors_table.verticalHeader().restoreGeometry(self.settings.value("sensorsVerticalHeader", QByteArray()))
        self.settings.endGroup()
        self.settings.beginGroup("state")
        self.restoreState(cast("QByteArray", self.settings.value("window", QByteArray())))
        self.sensors_table.horizontalHeader().restoreState(self.settings.value("sensorsHorizontalHeader", QByteArray()))
        self.sensors_table.verticalHeader().restoreState(self.settings.value("sensorsVerticalHeader", QByteArray()))
        self.settings.endGroup()

    def save_settings(self) -> None:
        self.settings.beginGroup("geometry")
        self.settings.setValue("window", self.saveGeometry())
        self.settings.setValue("sensors", self.sensors_table.saveGeometry())
        self.settings.setValue("sensorsHorizontalHeader", self.sensors_table.horizontalHeader().saveGeometry())
        self.settings.setValue("sensorsVerticalHeader", self.sensors_table.verticalHeader().saveGeometry())
        self.settings.endGroup()
        self.settings.beginGroup("state")
        self.settings.setValue("window", self.saveState())
        self.settings.setValue("sensorsHorizontalHeader", self.sensors_table.horizontalHeader().saveState())
        self.settings.setValue("sensorsVerticalHeader", self.sensors_table.verticalHeader().saveState())
        self.settings.endGroup()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.worker.requestInterruption()
        if self.worker.isRunning():
            event.ignore()
        else:
            self.save_settings()
            super().closeEvent(event)

    @Slot(dict)
    def on_status_received(self, data: dict[str, bool | None]) -> None:
        self.status_panel.display(data)

    @Slot(dict)
    def on_pressures_received(self, data: dict[str, bool | None]) -> None:
        self.pressures_panel.display(data)

    @Slot(list, dict)
    def on_thermometry_received(
        self,
        sensors: list[dict[str, float | str | bool | datetime]],
        heaters: dict[str, bool | None],
    ) -> None:
        self.heaters_panel.display(heaters)
        self.sensors_table.display(sensors)


if __name__ == "__main__":
    app: QApplication = QApplication(sys.argv)
    w: MainWindow = MainWindow()
    w.show()
    app.exec()
