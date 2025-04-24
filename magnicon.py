import configparser
import string
import sys
from ast import literal_eval
from collections.abc import Callable
from functools import partial
from os import PathLike
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, ClassVar, Final, NamedTuple, TextIO

import numpy as np
from nidaqmx.constants import WAIT_INFINITELY, AcquisitionType
from nidaqmx.system.system import System
from nidaqmx.task import Task
from numpy.typing import NDArray
from pyqtgraph import DateAxisItem, GraphicsLayoutWidget, PlotDataItem, PlotItem, PlotWidget
from qtpy.QtCore import QDateTime, Qt, QThread, Signal, Slot
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton, QTabWidget, QVBoxLayout, QWidget
from scipy.optimize import curve_fit
from scipy.signal import welch, windows

if TYPE_CHECKING:
    from nidaqmx.system.device import Device
    from nidaqmx.system.physical_channel import PhysicalChannel

ORIG_CONFIG_PATH: Path = Path(r"C:\MagniconData\TempViewerData\TempViewer_Config.cfg")
CONFIG_PATH: Path = (
    # try getting the config from the original app, fall back to a local file
    ORIG_CONFIG_PATH if sys.platform == "win32" and ORIG_CONFIG_PATH.exists() else Path("TempViewer_Config.cfg")
)


class ConfigParser(configparser.ConfigParser):
    _sentinel: ClassVar[object] = object()

    def getstr[**P](
        self,
        section: str,
        option: str,
        *,
        fallback: str | object = _sentinel,
        **kwargs: P.kwargs,
    ) -> str:
        try:
            data: str = self.get(section, option, **kwargs)
        except LookupError:
            if fallback is not ConfigParser._sentinel:
                return fallback
            raise
        else:
            if data.startswith('"') and data.endswith('"'):
                return literal_eval(data)
            msg: str = "The requested value is not a string"
            raise TypeError(msg)

    def getpath[**P](
        self,
        section: str,
        option: str,
        *,
        fallback: str | object = _sentinel,
        **kwargs: P.kwargs,
    ) -> Path:
        def normalize_path(path: str) -> Path:
            if path.startswith("/") and sys.platform == "win32":
                path = path.removeprefix("/")
                drive: str = path.partition("/")[0]
                if len(drive) == 1 and drive in string.ascii_letters:
                    path = ":".join((drive, path[1:]))
            return Path(path)

        data: str = self.getstr(section, option, fallback=fallback, **kwargs)
        if data.startswith("/"):
            return normalize_path(data)
        msg: str = "The requested value is not a string"
        raise TypeError(msg)

    def set[T](self, section: str, option: str, value: T | None = None) -> None:
        if isinstance(value, PureWindowsPath):
            value = "/".join(("", value.drive.rstrip(":"), value.parts[1:]))
        elif isinstance(value, PurePosixPath):
            value = value.as_posix()
        # not `elif` here intentionally
        if isinstance(value, str):
            value = '"' + value.replace("\\", r"\\").replace('"', r"\"") + '"'
        elif isinstance(value, bool):
            value = str(value).upper()
        super().set(section, option, value)


class Band(NamedTuple):
    lower: float
    upper: float


class NoiseTrend(NamedTuple):
    data: NDArray[np.float64]
    fs: float


class PSD(NamedTuple):
    frequency: NDArray[np.float64]
    pn_xx: NDArray[np.float64]


class FitResult(NamedTuple):
    s0: float | np.float64
    fc: float | np.float64
    p1: float | np.float64
    p2: float | np.float64
    s0_err: float | np.float64
    fc_err: float | np.float64
    p1_err: float | np.float64
    p2_err: float | np.float64
    avg: int
    band: Band
    time: QDateTime = QDateTime.currentDateTime()


class MagniCon:
    fft_windows: ClassVar[list[tuple[str, str | Callable[[int], NDArray[np.float64]] | None]]] = [
        ("Rectangle", windows.boxcar),
        ("Hann", windows.hann),
        ("Hamming", windows.hamming),
        # https://www.ni.com/docs/en-US/bundle/ni-rfsa-c-api-ref/page/rfsacref/nirfsa_attr_fft_window_type.html
        (
            "Blackman-Harris",
            partial(
                windows.general_cosine,
                a=[
                    0.42323,
                    -0.49755,
                    0.07922,
                ],
            ),
        ),
        # https://www.ni.com/docs/en-US/bundle/ni-rfsa-c-api-ref/page/rfsacref/nirfsa_attr_fft_window_type.html
        (
            "Exact Blackman",
            partial(
                windows.general_cosine,
                a=[
                    7938 / 18608,
                    -9240 / 18608,
                    1430 / 18608,
                ],
            ),
        ),
        ("Blackman", windows.blackman),  # an approximation of the “exact” window
        ("Flattop", windows.flattop),
        # https://www.ni.com/docs/en-US/bundle/ni-rfsa-c-api-ref/page/rfsacref/nirfsa_attr_fft_window_type.html
        (
            "4-Term B-H",
            partial(
                windows.general_cosine,
                a=[
                    0.35875,
                    -0.48829,
                    0.14128,
                    -0.01168,
                ],
            ),
        ),
        # https://dsp.stackexchange.com/a/51101
        # https://www.ni.com/docs/en-US/bundle/ni-rfsa-c-api-ref/page/rfsacref/nirfsa_attr_fft_window_type.html
        (
            "7-Term B-H",
            partial(
                windows.general_cosine,
                a=[
                    0.27105140069342,
                    -0.43329793923448,
                    0.21812299954311,
                    0.06592544638803,
                    0.01081174209837,
                    -0.00077658482522,
                    0.00001388721735,
                ],
            ),
        ),
        # https://www.ni.com/docs/en-US/bundle/ni-rfsa-c-api-ref/page/rfsacref/nirfsa_attr_fft_window_type.html
        (
            "Low Sidelobe",
            partial(
                windows.general_cosine,
                a=[
                    0.323215218,
                    -0.471492057,
                    0.17553428,
                    -0.028497078,
                    0.001261367,
                ],
            ),
        ),
        # https://www.recordingblogs.com/wiki/blackman-nuttall-window
        (
            "Blackman-Nuttall",
            partial(
                windows.general_cosine,
                a=[
                    0.3635819,
                    0.4891775,
                    0.1365995,
                    0.0106411,
                ],
            ),
        ),
        ("Triangle", windows.triang),
        ("Bartlett-Hann", windows.barthann),
        ("Bohman", windows.bohman),
        ("Parzen", windows.parzen),
        ("Welch", None),  # ???
        ("Kaiser", windows.kaiser),  # needs beta
        ("Dolph-Chebychev", windows.chebwin),  # needs attenuation
        ("Gaussian", windows.gaussian),  # needs standard deviation, σ ≤ 0.5  # noqa: RUF003
    ]

    # Averaging Mode:
    #  0 = No averaging
    #  1 = Vector averaging
    #  2 = RMS averaging
    #  3 = Peak hold

    # Weighting Mode:
    #  0 = Linear
    #  1 = Exponential

    def __init__(self) -> None:
        config: ConfigParser = ConfigParser()

        config.read(CONFIG_PATH)
        self.averaging: int = config.getint("FFT", "Avg Meas", fallback=80)
        # The two following config values are currently ignored:
        self.averaging_mode: int = config.getint("FFT", "Averaging Mode", fallback=2)
        self.weighting_mode: int = config.getint("FFT", "Weighting Mode", fallback=0)
        self.fft_window: int = config.getint("FFT", "Window", fallback=1)
        self.cal_avg: int = config.getint("FFT", "Avg Cal", fallback=5000)

        self.reject_plf: bool = config.getboolean("General Fit Settings", "PLF Reject")
        self.plf: float = config.getfloat("General Fit Settings", "PWR Line Freq", fallback=50.0)
        self.plf_span: float = config.getfloat("General Fit Settings", "Frequency Span", fallback=20.0)
        self.lpf: bool = config.getboolean("General Fit Settings", "LPF", fallback=True)
        cal_path: Path = config.getpath("General Fit Settings", "Cal Filepath")
        self.higher_cutoff: float = config.getfloat("General Fit Settings", "Higher Cutoff Measure", fallback=5e3)
        self.lower_cutoff: float = config.getfloat("General Fit Settings", "Lower Cutoff", fallback=50.0)
        self.rejected_bands: list[Band] = []
        reject_peaks: bool = config.getboolean("General Fit Settings", "Peak Reject")
        if reject_peaks:
            peak_path: Path = config.getpath("General Fit Settings", "Peak Filepath")
            for line in peak_path.read_text().splitlines():
                if line:
                    words: list[str] = line.split()
                    if len(words) == 2:  # noqa: PLR2004
                        self.rejected_bands.append(Band(float(words[0]), float(words[1])))

        system: Final[System] = System.local()
        device_name: str = config.getstr("DAQ", "DAQDeviceName", fallback="Dev1")
        try:
            self.squid: Final[Device] = system.devices[device_name]
        except LookupError:
            msg: str = f"Device {device_name!r} is not present in the system"
            raise RuntimeError(msg) from None
        self.channel: Final[PhysicalChannel] = self.squid.ai_physical_chans["ai7"]
        self.set_lpf(on=self.lpf)

        config.read(cal_path)
        self.t_ref: float = config.getfloat("Fit Parameters", "Tref")
        self.s0: float = config.getfloat("Fit Parameters", "S0(T)")
        self.fc: float = config.getfloat("Fit Parameters", "fc")
        self.p1: float = config.getfloat("Fit Parameters", "p1")
        self.p2: float = config.getfloat("Fit Parameters", "p2")

        self._noise_trend: NoiseTrend | None = None
        self._psd: PSD | None = None

        # /port1/line2 = ?, initially False
        # /port1/line3 = ?, initially True

    @staticmethod
    def _fit_s(f: NDArray[np.float64], s0: float, fc: float, p1: float, p2: float) -> NDArray[np.float64]:
        return s0 / (1.0 + (f / fc) ** (2.0 * p1)) ** p2

    def _measure_noise(
        self,
        *,
        averaging: int | None = None,
        resolution: float = 5.0,
        rate: float | None = None,
    ) -> PSD:
        if averaging is None:
            averaging = self.averaging
        window: str | None
        try:
            window = MagniCon.fft_windows[self.fft_window][1]
        except IndexError:
            window = None

        if averaging < 1:
            msg: str = "Averaging must be a positive number"
            raise ValueError(msg)
        if window is None:
            window = "hann"

        task_adc: Task
        with Task() as task_adc:
            task_adc.ai_channels.add_ai_voltage_chan(self.channel.name)
            if rate is None:
                rate = task_adc.timing.samp_clk_max_rate

            averaging_shift: float = 1.0 / self.plf
            averaging_step: int = round(rate * averaging_shift)
            if averaging != 1 and averaging_step < 1:
                msg: str = f"Rate must be greater than {self.plf}"
                raise ValueError(msg)

            length: int = round(rate / resolution) + (averaging - 1) * averaging_step
            task_adc.timing.cfg_samp_clk_timing(rate=rate, sample_mode=AcquisitionType.CONTINUOUS)
            task_adc.start()
            data: NDArray[np.float64] = np.asarray(
                task_adc.read(length, timeout=WAIT_INFINITELY),
                dtype=np.float64,
            )
            task_adc.stop()

        # cache the result
        self._noise_trend = NoiseTrend(data=data, fs=rate)

        freq: NDArray[np.float64]
        pn_xx: NDArray[np.float64]
        nperseg: int
        if averaging == 1:
            nperseg = data.size
            if callable(window):
                # noinspection PyCallingNonCallable
                window = window(nperseg)
            freq, pn_xx = welch(data, fs=rate, window=window, nperseg=nperseg)
        else:
            nperseg = data.size - (averaging - 1) * averaging_step
            if callable(window):
                # noinspection PyCallingNonCallable
                window = window(nperseg)
            freq, pn_xx = welch(
                np.column_stack(
                    [
                        # slice the data into (possibly) overlapping parts
                        data[a * averaging_step : -((averaging - a - 1) * averaging_step) or None]
                        for a in range(averaging)
                    ],
                ),
                fs=rate,
                window=window,
                nperseg=nperseg,
                axis=0,
            )
            pn_xx = np.mean(pn_xx, axis=1)

        # cache the result
        self._psd = PSD(frequency=freq, pn_xx=pn_xx)
        return self._psd

    def _good_to_fit(self, freq: NDArray[np.float64], band: Band | None = None) -> NDArray[np.bool_]:
        if band is None:
            band = Band(self.lower_cutoff, self.higher_cutoff)

        to_fit: NDArray[np.bool_] = (freq >= band.lower) & (freq <= band.upper)

        # reject PLF
        if self.reject_plf:
            for i in range(int(freq[-1] // self.plf)):
                to_fit &= np.abs(freq - i * self.plf) >= self.plf_span

        # reject user-defined peaks
        for band in self.rejected_bands:
            to_fit &= np.logical_or(freq <= band.lower, freq >= band.upper)

        return to_fit

    def _measure_noise_to_fit(
        self,
        *,
        averaging: int | None = None,
        band: Band | None = None,
        resolution: float = 5.0,
    ) -> PSD:
        psd: PSD = self._measure_noise(averaging=averaging, resolution=resolution)
        to_fit: NDArray[np.bool_] = self._good_to_fit(psd.frequency, band=band)
        return PSD(frequency=psd.frequency[to_fit], pn_xx=psd.pn_xx[to_fit])

    def _calculate_temperature(self, psd: PSD) -> float:
        fit_shape: NDArray[np.float64] = self._fit_s(psd.frequency, 1.0, self.fc, self.p1, self.p2)
        s: np.float64 = np.mean(psd.pn_xx / fit_shape, dtype=np.float64)
        return self.t_ref * s / self.s0

    def plot_fft(
        self,
        averaging: int | None = None,
        band: Band | None = None,
        resolution: float = 5.0,
    ) -> None:
        from matplotlib import pyplot as plt

        psd: PSD = self._measure_noise(averaging=averaging, resolution=resolution)

        to_fit: NDArray[np.bool_] = self._good_to_fit(psd.frequency, band=band)

        fit_freq: NDArray[np.float64] = psd.frequency[to_fit]
        fit_pn_xx: NDArray[np.float64] = psd.pn_xx[to_fit]
        fit_shape: NDArray[np.float64] = self._fit_s(fit_freq, 1.0, self.fc, self.p1, self.p2)
        s: np.float64 = np.mean(fit_pn_xx / fit_shape, dtype=np.float64)

        plt.loglog(psd.frequency, psd.pn_xx)
        plt.loglog(fit_freq, s * fit_shape)
        plt.show()

    def measure_temperature(self) -> float:
        return self._calculate_temperature(self._measure_noise_to_fit())

    def fit_params(self, psd: PSD, avg: int, band: Band) -> FitResult:
        log_f: NDArray[np.float64] = np.log(psd.frequency)
        log_s: NDArray[np.float64] = np.log(psd.pn_xx)

        fit: np.polynomial.Polynomial
        fit_s0: float
        fit_fc: float
        fit_p1: float
        fit_p2: float

        fit = np.polynomial.Polynomial.fit(log_f, log_s, deg=1, w=1 / psd.frequency).convert()
        fit_log_s0 = fit.coef[0]
        fit_s0 = np.exp(fit_log_s0)

        fit = np.polynomial.Polynomial.fit(log_f, log_s - fit_log_s0, deg=1, w=psd.frequency).convert()
        fit_fc = np.exp(-fit.coef[0] / fit.coef[1])
        fit_p1 = fit_p2 = np.sqrt(-0.5 * fit.coef[1])

        # noinspection PyTupleAssignmentBalance
        (fit_s0, fit_fc, fit_p1, fit_p2), p_cov = curve_fit(
            f=self._fit_s,
            xdata=psd.frequency,
            ydata=psd.pn_xx,
            p0=(fit_s0, fit_fc, fit_p1, fit_p2),
            bounds=(
                [0.5 * fit_s0, 0.5 * fit_fc, 0.5 * fit_p1, 0.5 * fit_p2],
                [2.0 * fit_s0, 2.0 * fit_fc, 2.0 * fit_p1, 2.0 * fit_p2],
            ),
        )

        return FitResult(fit_s0, fit_fc, fit_p1, fit_p2, *np.sqrt(np.diag(p_cov)), avg=avg, band=band)

    def write_cal(  # noqa: PLR0913
        self,
        filename: int | str | bytes | PathLike[str] | PathLike[bytes],
        t_ref: float,
        dt_ref: float,
        *,
        psd: PSD | None = None,
        avg: int | None = None,
        band: Band | None = None,
    ) -> None:
        if psd is None:
            psd = self._measure_noise_to_fit()
        if avg is None:
            avg = self.cal_avg
        if band is None:
            band = Band(self.lower_cutoff, self.higher_cutoff)

        fit_result: FitResult = self.fit_params(psd=psd, avg=avg, band=band)
        config: ConfigParser = ConfigParser()
        config.set("Fit Parameters", "Date_Time", fit_result.time.toString("yyyy-MM-dd_HH-mm-ss"))
        config.set("Fit Parameters", "Tref", t_ref)
        config.set("Fit Parameters", "DTref", dt_ref)
        config.set("Fit Parameters", "S0(T)", fit_result.s0)
        config.set("Fit Parameters", "fc", fit_result.fc)
        config.set("Fit Parameters", "p1", fit_result.p1)
        config.set("Fit Parameters", "p2", fit_result.p2)
        config.set("Fit Parameters", "std.dev. S0", fit_result.s0_err)
        config.set("Fit Parameters", "std.dev. fc", fit_result.fc_err)
        config.set("Fit Parameters", "std.dev. p1", fit_result.p1_err)
        config.set("Fit Parameters", "std.dev. p2", fit_result.p2_err)
        config.set("Fit Parameters", "Averages", fit_result.avg)
        config.set("Fit Parameters", "low f cal", fit_result.band.lower)
        config.set("Fit Parameters", "up f cal", fit_result.band.upper)
        config.set("Info", "DAQBox#", f"{self.squid.serial_num:x}")
        f_out: TextIO
        with open(filename, "w") as f_out:  # noqa: PTH123
            config.write(f_out)

    def set_lpf(self, *, on: bool) -> None:
        # /port1/line1 = LPF: True = off, False = on
        with Task() as task:
            task.do_channels.add_do_chan(self.squid.do_lines[1].name)
            task.write([not on], auto_start=True)
            task.wait_until_done()
            task.stop()

    @property
    def noise_trend(self) -> NoiseTrend:
        return self._noise_trend

    @property
    def psd(self) -> PSD:
        return self._psd


class QMagniCon(QThread, MagniCon):
    measured: Signal = Signal(float)

    def run(self) -> None:
        while not self.isInterruptionRequested():
            t: float = self.measure_temperature()
            self.measured.emit(t)


class MeasureUI(QWidget):
    def __init__(self, m: QMagniCon, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.m: QMagniCon = m
        self.m.started.connect(self.on_thread_started)
        self.m.finished.connect(self.on_thread_finished)
        self.m.measured.connect(self.show_result)

        main_layout: QVBoxLayout = QVBoxLayout(self)
        top_layout: QHBoxLayout = QHBoxLayout()
        self.result_label: QLabel = QLabel(self)
        self.button_start: QPushButton = QPushButton(self.tr("Start"), self)
        self.button_stop: QPushButton = QPushButton(self.tr("Stop"), self)
        plot_widget: PlotWidget = PlotWidget(self)
        plot_item: PlotItem = plot_widget.plotItem
        self.plot_line: PlotDataItem = plot_item.plot(symbolBrush="red", symbolPen="red")
        top_layout.addWidget(self.button_start)
        top_layout.addWidget(self.button_stop)
        top_layout.addWidget(self.result_label)
        main_layout.addLayout(top_layout, 0)
        main_layout.addWidget(plot_widget, 1)

        self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.button_stop.setEnabled(False)
        plot_item.setAxisItems({"bottom": DateAxisItem()})
        plot_item.setLabel(
            axis="left",
            text=self.tr("Temperature"),
            units=self.tr("K"),
        )

        self.button_start.clicked.connect(self.on_button_start_clicked)
        self.button_stop.clicked.connect(self.on_button_stop_clicked)

    @Slot(float)
    def show_result(self, result: float) -> None:
        self.result_label.setText(f"T = {result * 1e3:.3f} mK")
        x: NDArray[np.float64] | None
        y: NDArray[np.float64] | None
        x, y = self.plot_line.getOriginalDataset()
        if x is None:
            x = np.asarray([QDateTime.currentMSecsSinceEpoch() / 1000])
        else:
            x = np.append(x, QDateTime.currentMSecsSinceEpoch() / 1000)
        if y is None:  # noqa: SIM108
            y = np.asarray([result])
        else:
            y = np.append(y, result)
        self.plot_line.setData(x, y)

    @Slot()
    def on_thread_started(self) -> None:
        self.button_start.setEnabled(False)
        self.button_stop.setEnabled(True)

    @Slot()
    def on_thread_finished(self) -> None:
        self.button_stop.setEnabled(False)
        self.button_start.setEnabled(True)

    @Slot()
    def on_button_start_clicked(self) -> None:
        if not self.m.isRunning():
            self.button_start.setEnabled(False)
            self.m.start()

    @Slot()
    def on_button_stop_clicked(self) -> None:
        if self.m.isRunning():
            self.button_stop.setEnabled(False)
            self.m.terminate()
            self.m.wait()


class PreviewUI(GraphicsLayoutWidget):
    def __init__(self, m: QMagniCon, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.m: QMagniCon = m
        self.m.measured.connect(self.show_result)

        plot_item_noise: PlotItem = self.ci.addPlot(row=0, col=0)
        plot_item_spectrum: PlotItem = self.ci.addPlot(row=1, col=0)
        self.plot_line_noise: PlotDataItem = plot_item_noise.plot()
        self.plot_line_spectrum: PlotDataItem = plot_item_spectrum.plot()

        plot_item_noise.setLabel(axis="bottom", text=self.tr("Time"), units=self.tr("s"))
        plot_item_noise.setLabel(axis="left", text=self.tr("Voltage"), units=self.tr("V"))
        plot_item_spectrum.setLabel(axis="bottom", text=self.tr("Frequency"), units=self.tr("Hz"))
        plot_item_spectrum.setLabel(axis="left", text=self.tr("PSD"), units=self.tr("V/√︤㎐︦"))
        plot_item_spectrum.setLogMode(x=True, y=True)

    @Slot(float)
    def show_result(self, _: float) -> None:
        noise_trend: NoiseTrend = self.m.noise_trend
        psd: PSD = self.m.psd
        self.plot_line_noise.setData(
            np.linspace(
                start=0.0,
                stop=noise_trend.data.size / noise_trend.fs,
                num=noise_trend.data.size,
                dtype=np.float64,
            ),
            noise_trend.data,
        )
        self.plot_line_spectrum.setData(psd.frequency, psd.pn_xx)


class UI(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        central_widget: QTabWidget = QTabWidget(self)
        self.setCentralWidget(central_widget)

        self.m: QMagniCon = QMagniCon(self)

        central_widget.addTab(MeasureUI(self.m), self.tr("Measure"))
        central_widget.addTab(PreviewUI(self.m), self.tr("Preview"))

        config: ConfigParser = ConfigParser()
        config.read(CONFIG_PATH)
        try:
            left: int = config.getint("FrontPanel", "left")
            top: int = config.getint("FrontPanel", "top")
            right: int = config.getint("FrontPanel", "right")
            bottom: int = config.getint("FrontPanel", "bottom")
        except LookupError:
            pass
        else:
            self.setGeometry(left, top, right - left, bottom - top)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self.m.isRunning():
            self.m.terminate()
            self.m.wait()
        return super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = UI()
    w.show()
    app.exec()
