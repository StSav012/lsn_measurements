import pyqtgraph as pg
from qtpy.QtCore import QSettings, Qt, Slot
from qtpy.QtGui import QCloseEvent, QIcon
from qtpy.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.histogram import Histogram
from ui.safe_button import SafeButton

__all__ = ["LifetimeGUI"]


class LifetimeGUI(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        self.settings: QSettings = QSettings("SavSoft", "Lifetime", self)

        self.setWindowTitle(self.tr("Lifetime"))
        self.setWindowIcon(QIcon("lsn.svg"))

        self.central_widget: QWidget = QWidget(self)
        self.main_layout: QHBoxLayout = QHBoxLayout(self.central_widget)
        self.controls_layout: QVBoxLayout = QVBoxLayout()
        self.parameters_box: QGroupBox = QGroupBox(self.central_widget)
        self.parameters_layout: QFormLayout = QFormLayout(self.parameters_box)
        self.button_topmost: QPushButton = QPushButton(self.central_widget)
        self.button_drop_measurement: SafeButton = SafeButton("", timeout=4.0, parent=self.central_widget)
        self.histogram: Histogram = Histogram(self.central_widget)
        self.stop_sings_box: QGroupBox = QGroupBox(self.central_widget)
        self.stop_sings_box.setLayout(QVBoxLayout())
        self.buttons_layout: QHBoxLayout = QHBoxLayout()

        self.figure: pg.PlotWidget = pg.PlotWidget(self.central_widget)
        self.plot_lines: dict[int, pg.PlotDataItem] = {}
        self.figure.addLegend(offset=(30, -30))

        self.label_spent_time: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_bias: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_frequency: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_power: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_setting_time: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_delay_between_cycles: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_aux_voltage: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_temperature: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_loop_number: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_mean_lifetime: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_lifetime_std: pg.ValueLabel = pg.ValueLabel(self.central_widget)
        self.label_lifetime_mean_std_ratio: pg.ValueLabel = pg.ValueLabel(self.central_widget)

        self.stop_key_bias: QPushButton = QPushButton(self.stop_sings_box)
        self.stop_key_power: QPushButton = QPushButton(self.stop_sings_box)
        self.stop_key_frequency: QPushButton = QPushButton(self.stop_sings_box)
        self.stop_key_setting_time: QPushButton = QPushButton(self.stop_sings_box)
        self.stop_key_delay_between_cycles: QPushButton = QPushButton(self.stop_sings_box)
        self.stop_key_aux_voltage: QPushButton = QPushButton(self.stop_sings_box)
        self.stop_key_temperature: QPushButton = QPushButton(self.stop_sings_box)

        self.button_start: QPushButton = QPushButton(self.central_widget)
        self.button_pause: QPushButton = QPushButton(self.central_widget)
        self.button_stop: QPushButton = QPushButton(self.central_widget)

        self.setup_ui_appearance()
        self.load_settings()
        self.setup_actions()

    def setup_ui_appearance(self) -> None:
        x_axis: pg.AxisItem = self.figure.getAxis("bottom")
        x_axis.enableAutoSIPrefix(enable=False)
        y_axis: pg.AxisItem = self.figure.getAxis("left")
        y_axis.setLabel(text=self.tr("Lifetime"), units=self.tr("s"))
        y_axis.enableAutoSIPrefix(enable=False)
        self.figure.plotItem.ctrl.averageGroup.setChecked(False)
        self.figure.plotItem.setLogMode(x=False, y=True)
        self.figure.plotItem.showGrid(x=True, y=True)

        self.label_spent_time.suffix = self.tr("s")
        self.label_spent_time.formatStr = "{value:0.5f} {suffix}"
        self.label_bias.suffix = self.tr("nA")
        self.label_bias.formatStr = "{value:0.2f} {suffix}"
        self.label_frequency.suffix = self.tr("GHz")
        self.label_frequency.formatStr = "{value:0.4f} {suffix}"
        self.label_power.suffix = self.tr("dBm")
        self.label_setting_time.suffix = self.tr("ms")
        self.label_setting_time.formatStr = "{value:0.1f} {suffix}"
        self.label_delay_between_cycles.suffix = self.tr("ms")
        self.label_delay_between_cycles.formatStr = "{value:0.1f} {suffix}"
        self.label_aux_voltage.suffix = self.tr("mV")
        self.label_aux_voltage.formatStr = "{value:0.2f} {suffix}"
        self.label_temperature.suffix = self.tr("mK")
        self.label_temperature.formatStr = "{value:0.2f} {suffix}"
        self.label_loop_number.formatStr = "{value:.0f}"
        self.label_mean_lifetime.suffix = self.tr("s")
        self.label_mean_lifetime.formatStr = "{value:0.5f} {suffix}"
        self.label_lifetime_std.suffix = self.tr("s")
        self.label_lifetime_std.formatStr = "{value:0.5f} {suffix}"
        self.label_lifetime_mean_std_ratio.formatStr = "{value:0.5f}"

        self.figure.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self.main_layout.addWidget(self.figure)
        self.main_layout.addLayout(self.controls_layout)
        self.main_layout.setStretch(0, 1)
        self.controls_layout.addWidget(self.parameters_box)
        self.controls_layout.addWidget(self.button_topmost)
        self.controls_layout.addWidget(self.button_drop_measurement)
        self.controls_layout.addWidget(self.histogram)
        self.controls_layout.addWidget(self.stop_sings_box)
        self.controls_layout.addLayout(self.buttons_layout)

        self.parameters_layout.addRow(self.tr("Since bias set:"), self.label_spent_time)
        self.parameters_layout.addRow(self.tr("Bias current:"), self.label_bias)
        self.parameters_layout.addRow(self.tr("Frequency:"), self.label_frequency)
        self.parameters_layout.addRow(self.tr("Power:"), self.label_power)
        self.parameters_layout.addRow(self.tr("Setting time:"), self.label_setting_time)
        self.parameters_layout.addRow(self.tr("Delay b/w cycles:"), self.label_delay_between_cycles)
        self.parameters_layout.addRow(self.tr("Aux voltage:"), self.label_aux_voltage)
        self.parameters_layout.addRow(self.tr("Temperature:"), self.label_temperature)
        self.parameters_layout.addRow(self.tr("Loop number:"), self.label_loop_number)
        self.parameters_layout.addRow(self.tr("Mean lifetime:"), self.label_mean_lifetime)
        self.parameters_layout.addRow(self.tr("Lifetime StD:"), self.label_lifetime_std)
        self.parameters_layout.addRow(self.tr("Mean to StD ratio:"), self.label_lifetime_mean_std_ratio)

        self.button_topmost.setText(self.tr("Keep the Window Topmost"))
        self.button_topmost.setCheckable(True)

        self.button_drop_measurement.setText(self.tr("Next Measurement"))

        self.histogram.set_label(text=self.tr("Lifetime"), unit=self.tr("s"))
        self.histogram.setLogMode(x=True, y=True)

        self.stop_key_bias.setText(self.tr("Stop after this Bias"))
        self.stop_key_power.setText(self.tr("Stop after this Power"))
        self.stop_key_frequency.setText(self.tr("Stop after this Frequency"))
        self.stop_key_setting_time.setText(self.tr("Stop after this Setting Time"))
        self.stop_key_delay_between_cycles.setText(self.tr("Stop after this Delay"))
        self.stop_key_aux_voltage.setText(self.tr("Stop after this Aux Voltage"))
        self.stop_key_temperature.setText(self.tr("Stop after this Temperature"))

        self.stop_key_bias.setCheckable(True)
        self.stop_key_power.setCheckable(True)
        self.stop_key_frequency.setCheckable(True)
        self.stop_key_setting_time.setCheckable(True)
        self.stop_key_delay_between_cycles.setCheckable(True)
        self.stop_key_aux_voltage.setCheckable(True)
        self.stop_key_temperature.setCheckable(True)

        self.stop_sings_box.layout().addWidget(self.stop_key_bias)
        self.stop_sings_box.layout().addWidget(self.stop_key_power)
        self.stop_sings_box.layout().addWidget(self.stop_key_frequency)
        self.stop_sings_box.layout().addWidget(self.stop_key_setting_time)
        self.stop_sings_box.layout().addWidget(self.stop_key_delay_between_cycles)
        self.stop_sings_box.layout().addWidget(self.stop_key_aux_voltage)
        self.stop_sings_box.layout().addWidget(self.stop_key_temperature)

        self.buttons_layout.addWidget(self.button_start)
        self.buttons_layout.addWidget(self.button_pause)
        self.buttons_layout.addWidget(self.button_stop)

        self.button_start.setText(self.tr("Start"))
        self.button_pause.setText(self.tr("Pause"))
        self.button_pause.setCheckable(True)
        self.button_stop.setText(self.tr("Stop"))
        self.button_stop.setDisabled(True)

        self.setCentralWidget(self.central_widget)

    def setup_actions(self) -> None:
        self.button_topmost.toggled.connect(self.on_button_topmost_toggled)
        self.button_start.clicked.connect(self.on_button_start_clicked)
        self.button_stop.clicked.connect(self.on_button_stop_clicked)

    def load_settings(self) -> None:
        self.restoreGeometry(self.settings.value("windowGeometry", b""))
        self.restoreState(self.settings.value("windowState", b""))

    def save_settings(self) -> None:
        self.settings.setValue("windowGeometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.sync()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.on_button_stop_clicked()
        self.save_settings()
        event.accept()

    @Slot(bool)
    def on_button_topmost_toggled(self, on: bool) -> None:
        if on:
            self.setWindowFlags(
                self.windowFlags() | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowStaysOnTopHint,
            )
            self.show()
        else:
            self.setWindowFlags(
                self.windowFlags() ^ (Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowStaysOnTopHint),
            )
            self.show()

    @Slot()
    def on_button_start_clicked(self) -> None:
        self.button_start.setDisabled(True)
        self.button_pause.setChecked(False)
        self.button_stop.setEnabled(True)

    @Slot()
    def on_button_stop_clicked(self) -> None:
        self.button_stop.setDisabled(True)
        self.button_start.setEnabled(True)
