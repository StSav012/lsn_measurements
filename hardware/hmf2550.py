from typing import ClassVar, Final

try:
    from .scpi_device import SCPIDevice, SCPIDeviceSubCategory
except ImportError:
    from scpi_device import SCPIDevice, SCPIDeviceSubCategory


__all__ = ["HMF2550"]


class HMF2550(SCPIDevice):
    _PORT: Final[int] = 9111

    frequency: property = SCPIDevice.property_by_command(
        "frequency",
        (float, "minimum", "maximum"),
        doc="The output frequency.",
    )

    class _Burst(SCPIDeviceSubCategory):
        prefix: ClassVar[str] = "burst"

        mode: property = SCPIDeviceSubCategory.subproperty_by_command(
            "mode",
            ("triggered", "gated"),
            doc="""\
    The burst mode.
            
    TRIGgered:	A trigger will generate a burst with a predefined number of cycles.
    GATed:		The signal will be either on or off, depending on the level of the external signal
                at the “Trigger input/ouput” connector. If the gate signal is “true” (+5 V, TTL high),
                the function generator will deliver a continuous signal until the “gate” closes (0 V, TTL low).
                If no power is applied to the TRIG INPUT connector, the output signal
                will stop as the function generator will stop generation.""",
        )
        n_cycles: property = SCPIDeviceSubCategory.subproperty_by_command(
            "ncycles",
            int,
            doc="The number of burst cycles.",
        )
        state: property = SCPIDeviceSubCategory.subproperty_by_command(
            "state",
            bool,
            doc="The state of the burst function.",
        )

    class _Function(SCPIDeviceSubCategory):
        prefix: ClassVar[str] = "function"

        name: property = SCPIDeviceSubCategory.subproperty_by_command(
            "",
            ("sinusoid", "square", "ramp", "pulse", "arbitrary"),
            doc="The output function.",
        )

        class _Pulse(SCPIDeviceSubCategory):
            prefix: ClassVar[str] = "function:pulse"

            duty_cycle: property = SCPIDeviceSubCategory.subproperty_by_command(
                "dcycle", float, doc="The duty cycle of the pulse function."
            )
            edge_time: property = SCPIDeviceSubCategory.subproperty_by_command(
                "etime", float, doc="The edge time of the pulse function."
            )

            class _Width(SCPIDeviceSubCategory):
                prefix: ClassVar[str] = "function:pulse:width"

                high: property = SCPIDeviceSubCategory.subproperty_by_command(
                    "high",
                    (float, "minimum", "maximum"),
                    doc="The high width of the pulse function depending on the frequency setting.",
                )
                low: property = SCPIDeviceSubCategory.subproperty_by_command(
                    "low",
                    (float, "minimum", "maximum"),
                    doc="The low width of the pulse function depending on the frequency setting.",
                )

            def __init__(self, parent: SCPIDevice) -> None:
                super().__init__(parent)

                self.width: Final[HMF2550._Function._Pulse._Width] = HMF2550._Function._Pulse._Width(self.parent)

        def __init__(self, parent: SCPIDevice) -> None:
            super().__init__(parent)

            self.pulse: Final[HMF2550._Function._Pulse] = HMF2550._Function._Pulse(self.parent)

    class _Output(SCPIDeviceSubCategory):
        prefix: ClassVar[str] = "output"

        state: property = SCPIDeviceSubCategory.subproperty_by_command(
            "",
            bool,
            doc="The instrument output.",
        )
        load: property = SCPIDeviceSubCategory.subproperty_by_command(
            "load",
            (float, "terminated", "infinity"),
            doc="The instrument output load.",
        )

    class _Trigger(SCPIDeviceSubCategory):
        prefix: ClassVar[str] = "trigger"

        slope: property = SCPIDeviceSubCategory.subproperty_by_command(
            "slope",
            ("positive", "negative"),
            doc="Define the slope of the trigger input.",
        )
        source: property = SCPIDeviceSubCategory.subproperty_by_command(
            "source",
            ("immediate", "external"),
            doc="Define the trigger source.",
        )

        def immediate(self) -> None:
            """Trigger the device immediately if it is configured to wait for trigger events."""
            self.parent.communicate(":".join((HMF2550._Trigger.prefix, "immediate")))

    class _Voltage(SCPIDeviceSubCategory):
        prefix: ClassVar[str] = "voltage"

        high: property = SCPIDeviceSubCategory.subproperty_by_command(
            "high",
            (float, "minimum", "maximum"),
            doc="The high level voltage.",
        )
        low: property = SCPIDeviceSubCategory.subproperty_by_command(
            "low",
            (float, "minimum", "maximum"),
            doc="The low level voltage.",
        )
        offset: property = SCPIDeviceSubCategory.subproperty_by_command(
            "offset",
            (float, "minimum", "maximum"),
            doc="The output offset value.",
        )

    def __init__(self, ip: str | None = None, *, expected: bool = True) -> None:
        super().__init__(ip, HMF2550._PORT, terminator=b"\n", expected=expected, reset=False)

        self.burst: Final[HMF2550._Burst] = HMF2550._Burst(self)
        self.function: Final[HMF2550._Function] = HMF2550._Function(self)
        self.output: Final[HMF2550._Output] = HMF2550._Output(self)
        self.trigger: Final[HMF2550._Trigger] = HMF2550._Trigger(self)
        self.voltage: Final[HMF2550._Voltage] = HMF2550._Voltage(self)


if __name__ == "__main__":
    g: HMF2550 = HMF2550()
    print(f"{g.idn = !r}")
    print(f"{g.function.name = !r}")
    print(f"{g.output.state = !r}")
    print(f"{g.output.load = !r}")
    print(f"{g.voltage.high = !r}")
    print(f"{g.voltage.low = !r}")
    print(f"{g.function.pulse.width.high = !r}")
    print(f"{g.function.pulse.width.low = !r}")
    print(f"{g.function.pulse.duty_cycle = !r}")
    print(f"{g.function.pulse.edge_time = !r}")
    print(f"{g.burst.mode = !r}")
    print(f"{g.burst.n_cycles = !r}")
    print(f"{g.burst.state = !r}")
    print(f"{g.trigger.source = !r}")
    print(f"{g.trigger.slope = !r}")

    g.frequency = "max"
    print(f"{g.frequency = !r}")
    g.frequency = "min"
    print(f"{g.frequency = !r}")
    g.frequency = 1e5
    print(f"{g.frequency = !r}")

    g.voltage.low = "min"
    print(f"{g.voltage.low = !r}")
    g.voltage.low = 0.0
    print(f"{g.voltage.low = !r}")
