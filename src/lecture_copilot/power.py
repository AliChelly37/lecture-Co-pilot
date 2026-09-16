"""Power state (AC vs battery) and a sleep guard, both via Win32 and both
no-ops elsewhere. The model default depends on power; battery drain per
lecture-hour is a tracked metric.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class PowerState:
    on_ac: bool
    battery_percent: int | None

    @property
    def label(self) -> str:
        return "ac" if self.on_ac else "battery"


def read_power_state() -> PowerState:
    if sys.platform != "win32":
        return PowerState(on_ac=True, battery_percent=None)

    class SYSTEM_POWER_STATUS(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    status = SYSTEM_POWER_STATUS()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return PowerState(on_ac=True, battery_percent=None)
    percent = None if status.BatteryLifePercent == 255 else int(status.BatteryLifePercent)
    # ACLineStatus: 0 offline, 1 online, 255 unknown (treat unknown as AC).
    return PowerState(on_ac=status.ACLineStatus != 0, battery_percent=percent)


_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


class SleepGuard:
    """Keeps Windows from sleeping while a lecture is being recorded.

    Display sleep is still allowed; only system sleep is blocked. The guard is
    per-thread in Win32 terms, so acquire and release from the same thread.
    """

    def acquire(self) -> None:
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)

    def release(self) -> None:
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
