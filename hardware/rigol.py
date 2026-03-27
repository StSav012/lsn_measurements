import time
from typing import Final, Literal

try:
    from .scpi_device import SCPIDevice, SCPIDeviceSubCategory
except ImportError:
    from scpi_device import SCPIDevice, SCPIDeviceSubCategory

__all__ = ["DG1000Z"]


class DG1000Z(SCPIDevice):
    _PORT: Final[int] = 5555

    class _Output(SCPIDeviceSubCategory):
        impedance: property = SCPIDeviceSubCategory.subproperty_by_command(
            "impedance",
            (float, "infinity", "minimum", "maximum"),
            doc="The output impedance, same as load.",
        )
        load: property = SCPIDeviceSubCategory.subproperty_by_command(
            "load",
            (float, "infinity", "minimum", "maximum"),
            doc="The output load, same as impedance.",
        )
        mode: property = SCPIDeviceSubCategory.subproperty_by_command(
            "mode",
            ("normal", "gated"),
            doc="The output mode.",
        )
        polarity: property = SCPIDeviceSubCategory.subproperty_by_command(
            "polarity",
            ("normal", "inverted"),
            doc="The output polarity.",
        )
        state: property = SCPIDeviceSubCategory.subproperty_by_command(
            "state",
            bool,
            doc="The output state.",
        )

        class _Sync(SCPIDeviceSubCategory):
            sub_prefix = "sync"

            delay: property = SCPIDeviceSubCategory.subproperty_by_command(
                "state",
                (float, "minimum", "maximum"),
                doc="The sync signal output delay.",
            )
            polarity: property = SCPIDeviceSubCategory.subproperty_by_command(
                "polarity",
                ("normal", "inverted"),
                doc="The sync signal output polarity.",
            )
            state: property = SCPIDeviceSubCategory.subproperty_by_command(
                "state",
                bool,
                doc="The sync signal output state.",
            )

        def __init__(self, parent: SCPIDevice, channel: Literal[1, 2]) -> None:
            super().__init__(parent, prefix=f":output{channel}")

            self.sync: Final[DG1000Z._Output._Sync] = DG1000Z._Output._Sync(self.parent, prefix=self._prefix)

        def __bool__(self) -> bool:
            return self.state

    class _Source(SCPIDeviceSubCategory):
        class _Burst(SCPIDeviceSubCategory):
            sub_prefix = "burst"

            mode: property = SCPIDeviceSubCategory.subproperty_by_command(
                "mode",
                ("triggered", "infinity", "gated"),
                doc="""The burst type of the specified channel:
                       N-cycle (triggered), infinite (inifinity), or gated.""",
            )
            n_cycles: property = SCPIDeviceSubCategory.subproperty_by_command(
                "ncycles",
                (lambda _s: int(float(_s)), "maximum", "minimum"),
                doc="The number of cycles for the N-cycle burst.",
            )
            phase: property = SCPIDeviceSubCategory.subproperty_by_command(
                "phase", (float, "maximum", "minimum"), doc="The start phase of the burst function in degrees."
            )
            state: property = SCPIDeviceSubCategory.subproperty_by_command(
                "state", bool, doc="The state of the burst function."
            )
            t_delay: property = SCPIDeviceSubCategory.subproperty_by_command(
                "tdelay", (float, "maximum", "minimum"), doc="The delay of the burst function."
            )
            idle: property = SCPIDeviceSubCategory.subproperty_by_command(
                "idle",
                (int, "fpt", "top", "center", "bottom"),
                doc="""The idle level position of the burst mode.
                       The FPt stands for the first point.
                       The number may be from 1 to 16_383.""",
            )

            class _Trigger(SCPIDeviceSubCategory):
                sub_prefix = "trigger"

                slope: property = SCPIDeviceSubCategory.subproperty_by_command(
                    "slope",
                    ("positive", "negative"),
                )
                source: property = SCPIDeviceSubCategory.subproperty_by_command(
                    "source",
                    ("internal", "external", "manual"),
                )
                trig_out: property = SCPIDeviceSubCategory.subproperty_by_command(
                    "trigout",
                    ("positive", "negative", "off"),
                )

                def __call__(self) -> None:
                    self.issue("")

                def immediate(self) -> None:
                    self.issue("immediate")

            def __init__(self, parent: SCPIDevice, prefix: str) -> None:
                super().__init__(parent, prefix)

                self.trigger: Final[DG1000Z._Source._Burst._Trigger] = DG1000Z._Source._Burst._Trigger(
                    self.parent, prefix=self._prefix
                )

        class _Frequency(SCPIDeviceSubCategory):
            sub_prefix = "frequency"

            center: property = SCPIDeviceSubCategory.subproperty_by_command(
                "center",
                (float, "maximum", "minimum"),
                doc="The center frequency of the sweep function in Hz.",
            )
            fixed: property = SCPIDeviceSubCategory.subproperty_by_command(
                "fixed",
                (float, "maximum", "minimum"),
                doc="The frequency of basic waveforms and arbitrary waveform in Hz.",
            )
            span: property = SCPIDeviceSubCategory.subproperty_by_command(
                "span",
                (float, "maximum", "minimum"),
                doc="The frequency span of the sweep function in Hz.",
            )
            start: property = SCPIDeviceSubCategory.subproperty_by_command(
                "start",
                (float, "maximum", "minimum"),
                doc="The start frequency of the sweep function in Hz.",
            )
            stop: property = SCPIDeviceSubCategory.subproperty_by_command(
                "stop",
                (float, "maximum", "minimum"),
                doc="The stop frequency of the sweep function in Hz.",
            )

            class _Couple(SCPIDeviceSubCategory):
                sub_prefix = "couple"

                mode: property = SCPIDeviceSubCategory.subproperty_by_command(
                    "mode",
                    ("offset", "ratio"),
                    doc="The frequency coupling mode to frequency deviation (offset) or frequency ratio.",
                )
                offset: property = SCPIDeviceSubCategory.subproperty_by_command(
                    "offset", float, doc="The frequency deviation in the frequency coupling in Hz."
                )
                ratio: property = SCPIDeviceSubCategory.subproperty_by_command(
                    "ratio", float, doc="The frequency ratio in the frequency coupling."
                )
                state: property = SCPIDeviceSubCategory.subproperty_by_command("state", bool)

                def __bool__(self) -> bool:
                    return self.state

            def __init__(self, parent: SCPIDevice, prefix: str) -> None:
                super().__init__(parent, prefix)

                self.couple: Final[DG1000Z._Source._Frequency._Couple] = DG1000Z._Source._Frequency._Couple(
                    self.parent, prefix=self._prefix
                )

        class _Function(SCPIDeviceSubCategory):
            sub_prefix = "function"

            shape: property = SCPIDeviceSubCategory.subproperty_by_command(
                "shape",
                [
                    "sinusoid",
                    "square",
                    "ramp",
                    "pulse",
                    "noise",
                    "user",
                    "harmonic",
                    "dc",
                    "kaiser",
                    "roundpm",
                    "sinc",
                    "negramp",
                    "attalt",
                    "ampalt",
                    "stairdn",
                    "stairup",
                    "stairud",
                    "cpulse",
                    "ppulse",
                    "npulse",
                    "trapezia",
                    "roundhalf",
                    "abssine",
                    "abssinehalf",
                    "sinetra",
                    "sinever",
                    "exprise",
                    "expfall",
                    "tan",
                    "cot",
                    "sqrt",
                    "x2data",
                    "gauss",
                    "haversine",
                    "lorentz",
                    "dirichlet",
                    "gausspulse",
                    "airy",
                    "cardiac",
                    "quake",
                    "gamma",
                    "voice",
                    "tv",
                    "combin",
                    "bandlimited",
                    "stepresp",
                    "butterworth",
                    "chebyshev1",
                    "chebyshev2",
                    "boxcar",
                    "barlett",
                    "triang",
                    "blackman",
                    "hamming",
                    "hanning",
                    "dualtone",
                    "acos",
                    "acosh",
                    "acotcon",
                    "acotpro",
                    "acothcon",
                    "acothpro",
                    "acsccon",
                    "acscpro",
                    "acschcon",
                    "acschpro",
                    "aseccon",
                    "asecpro",
                    "asech",
                    "asin",
                    "asinh",
                    "atan",
                    "atanh",
                    "besselj",
                    "bessely",
                    "cauchy",
                    "cosh",
                    "cosint",
                    "cothcon",
                    "cothpro",
                    "csccon",
                    "cscpro",
                    "cschcon",
                    "cschpro",
                    "cubic",
                    "erf",
                    "erfc",
                    "erfcinv",
                    "erfinv",
                    "laguerre",
                    "laplace",
                    "legend",
                    "log",
                    "lognormal",
                    "maxwell",
                    "rayleigh",
                    "recipcon",
                    "recippro",
                    "seccon",
                    "secpro",
                    "sech",
                    "sinh",
                    "sinint",
                    "tanh",
                    "versiera",
                    "weibull",
                    "barthann",
                    "blackmanh",
                    "bohmanwin",
                    "chebwin",
                    "flattopwin",
                    "nuttallwin",
                    "parzenwin",
                    "taylorwin",
                    "tukeywin",
                    "cwpusle",
                    "lfpulse",
                    "lfmpulse",
                    "eog",
                    "eeg",
                    "emg",
                    "pulsilogram",
                    "tens1",
                    "tens2",
                    "tens3",
                    "surge",
                    "dampedosc",
                    "swingosc",
                    "radar",
                    "threeam",
                    "threefm",
                    "threepm",
                    "threepwm",
                    "threepfm",
                    "resspeed",
                    "mcnosie",
                    "pahcur",
                    "ripple",
                    "iso76372tp1",
                    "iso76372tp2a",
                    "iso76372tp2b",
                    "iso76372tp3a",
                    "iso76372tp3b",
                    "iso76372tp4",
                    "iso76372tp5a",
                    "iso76372tp5b",
                    "iso167502sp",
                    "iso167502vr",
                    "scr",
                    "ignition",
                    "nimhdischarge",
                    "gatevibr",
                ],
                first_matching=True,
            )

            class _Pulse(SCPIDeviceSubCategory):
                sub_prefix = "pulse"

                d_cycle: property = SCPIDeviceSubCategory.subproperty_by_command(
                    "dcycle", (float, "minimum", "maximum"), doc="The pulse duty cycle in percent."
                )
                hold: property = SCPIDeviceSubCategory.subproperty_by_command(
                    "hold",
                    ("width", "dcycle"),
                    doc="""The highlighted item of the specified channel:
                           pulse width or pulse duty cycle.""",
                )
                period: property = SCPIDeviceSubCategory.subproperty_by_command(
                    "period", (float, "minimum", "maximum"), doc="The pulse period in seconds."
                )
                symmetry: property = SCPIDeviceSubCategory.subproperty_by_command(
                    "symmetry", (float, "minimum", "maximum"), doc="The pulse symmetry in percent."
                )
                width: property = SCPIDeviceSubCategory.subproperty_by_command(
                    "width", (float, "minimum", "maximum"), doc="The pulse width in seconds."
                )

                class _Transition(SCPIDeviceSubCategory):
                    sub_prefix = "transition"

                    both: property = SCPIDeviceSubCategory.subproperty_by_command(
                        "both",
                        (float, "minimum", "maximum"),
                        doc="The pulse rise and fall durations in seconds.",
                        read=False,
                    )
                    leading: property = SCPIDeviceSubCategory.subproperty_by_command(
                        "leading",
                        (float, "minimum", "maximum"),
                        doc="The pulse rise duration in seconds.",
                    )
                    trailing: property = SCPIDeviceSubCategory.subproperty_by_command(
                        "trailing",
                        (float, "minimum", "maximum"),
                        doc="The pulse fall duration in seconds.",
                    )

                def __init__(self, parent: SCPIDevice, prefix: str) -> None:
                    super().__init__(parent, prefix)

                    self.transition: Final[DG1000Z._Source._Function._Pulse._Transition] = (
                        DG1000Z._Source._Function._Pulse._Transition(self.parent, prefix=self._prefix)
                    )

            def __init__(self, parent: SCPIDevice, prefix: str) -> None:
                super().__init__(parent, prefix)

                self.pulse: Final[DG1000Z._Source._Function._Pulse] = DG1000Z._Source._Function._Pulse(
                    self.parent, prefix=self._prefix
                )

        class _Pulse(SCPIDeviceSubCategory):
            sub_prefix = "pulse"

            d_cycle: property = SCPIDeviceSubCategory.subproperty_by_command(
                "dcycle",
                (float, "minimum", "maximum"),
                doc="The pulse duty cycle in percent.",
            )
            hold: property = SCPIDeviceSubCategory.subproperty_by_command(
                "hold",
                ("width", "dcycle"),
                doc="""The highlighted item of the specified channel:
                       pulse width or pulse duty cycle.""",
            )
            width: property = SCPIDeviceSubCategory.subproperty_by_command(
                "width",
                (float, "minimum", "maximum"),
                doc="The pulse width in seconds.",
            )

            class _Transition(SCPIDeviceSubCategory):
                sub_prefix = "transition"

                leading: property = SCPIDeviceSubCategory.subproperty_by_command(
                    "leading",
                    (float, "minimum", "maximum"),
                    doc="The pulse rise duration in seconds.",
                )
                trailing: property = SCPIDeviceSubCategory.subproperty_by_command(
                    "trailing",
                    (float, "minimum", "maximum"),
                    doc="The pulse fall duration in seconds.",
                )

            def __init__(self, parent: SCPIDevice, prefix: str) -> None:
                super().__init__(parent, prefix)

                self.transition: Final[DG1000Z._Source._Pulse._Transition] = DG1000Z._Source._Pulse._Transition(
                    self.parent, prefix=self._prefix
                )

        class _Voltage(SCPIDeviceSubCategory):
            sub_prefix = "voltage"

            amplitude: property = SCPIDeviceSubCategory.subproperty_by_command(
                "amplitude",
                (float, "minimum", "maximum"),
            )
            high: property = SCPIDeviceSubCategory.subproperty_by_command(
                "high",
                (float, "minimum", "maximum"),
            )
            low: property = SCPIDeviceSubCategory.subproperty_by_command(
                "low",
                (float, "minimum", "maximum"),
            )
            offset: property = SCPIDeviceSubCategory.subproperty_by_command(
                "offset",
                (float, "minimum", "maximum"),
            )
            unit: property = SCPIDeviceSubCategory.subproperty_by_command(
                "unit",
                ("vpp", "vrms", "dbm"),
            )

        def __init__(self, parent: SCPIDevice, channel: Literal[1, 2]) -> None:
            super().__init__(parent, prefix=f":source{channel}")

            self.burst: Final[DG1000Z._Source._Burst] = DG1000Z._Source._Burst(self.parent, prefix=self._prefix)
            self.frequency: Final[DG1000Z._Source._Frequency] = DG1000Z._Source._Frequency(
                self.parent, prefix=self._prefix
            )
            self.function: Final[DG1000Z._Source._Function] = DG1000Z._Source._Function(
                self.parent, prefix=self._prefix
            )
            self.pulse: Final[DG1000Z._Source._Pulse] = DG1000Z._Source._Pulse(self.parent, prefix=self._prefix)
            self.voltage: Final[DG1000Z._Source._Voltage] = DG1000Z._Source._Voltage(self.parent, prefix=self._prefix)

    def __init__(self, ip: str | None = None, reset: bool = True) -> None:
        super().__init__(ip, DG1000Z._PORT, expected=True, terminator=b"\n", reset=reset)
        self.output1: Final[DG1000Z._Output] = DG1000Z._Output(self, channel=1)
        self.source1: Final[DG1000Z._Source] = DG1000Z._Source(self, channel=1)
        self.output2: Final[DG1000Z._Output] = DG1000Z._Output(self, channel=2)
        self.source2: Final[DG1000Z._Source] = DG1000Z._Source(self, channel=2)


if __name__ == "__main__":
    s: DG1000Z = DG1000Z(reset=True)
    print(s.idn)

    print(f"{s.output1.state = }")
    print(f"{s.source1.function.shape = }")
    print(f"{s.output1.load = }")
    print(f"{s.source1.voltage.high = }")
    print(f"{s.source1.voltage.low = }")
    print(f"{s.source1.function.pulse.width = }")
    print(f"{s.source1.function.pulse.transition.leading = }")
    print(f"{s.source1.function.pulse.transition.trailing = }")
    print(f"{s.output1.state = }")

    s.output1.state = False
    s.output1.load = "infinity"
    s.source1.function.shape = "pulse"
    s.source1.function.pulse.period = 0.005
    s.source1.function.pulse.width = 0.0001
    s.source1.function.pulse.transition.both = "minimum"
    s.source1.voltage.high = 0.0
    s.source1.voltage.low = -5.0
    s.output1.sync.state = True
    time.sleep(1)
    s.output1.polarity = "inverted"
    s.output1.state = True

    print(s.idn)
    print(s.query(":system:error?"))
    print(f"{s.output1.state = }")
    print(f"{s.source1.function.shape = }")
    print(f"{s.output1.load = }")
    print(f"{s.source1.voltage.high = }")
    print(f"{s.source1.voltage.low = }")
    print(f"{s.source1.function.pulse.width = }")
    print(f"{s.source1.function.pulse.transition.leading = }")
    print(f"{s.source1.function.pulse.transition.trailing = }")
    print(f"{s.output1.state = }")
