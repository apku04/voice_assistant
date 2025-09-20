#!/usr/bin/env python3
from __future__ import annotations

import os
import signal
import sys
import time
import subprocess
from pathlib import Path

try:
    from gpiozero import LED, Device
    from gpiozero.pins.lgpio import LGPIOFactory
    Device.pin_factory = LGPIOFactory()  # Use lgpio on Pi 5
except Exception:
    LED = None  # type: ignore

PIN = 26  # GPIO26 (physical pin 37)
PID_FILE = Path("/tmp/va_fan.pid")


def _read_pid() -> int | None:
    try:
        pid = int(PID_FILE.read_text().strip())
        # Check if process exists
        os.kill(pid, 0)
        return pid
    except Exception:
        return None


def _write_pid(pid: int) -> None:
    try:
        PID_FILE.write_text(str(pid))
    except Exception:
        pass


def _clear_pid() -> None:
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except Exception:
        pass


def _hold_loop():
    if LED is None:
        print("fan lib not available")
        return 1
    fan = LED(PIN)
    # Turn on and hold until terminated
    fan.on()

    # Record PID for control
    _write_pid(os.getpid())

    stop = False

    def _sigterm(_signo, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    try:
        while not stop:
            time.sleep(0.5)
    finally:
        try:
            fan.off()
        except Exception:
            pass
        _clear_pid()
    return 0


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"on", "off", "status", "_hold"}:
        print("usage: fan_controller.py on|off|status")
        return 2

    action = sys.argv[1]

    if action == "_hold":
        return _hold_loop()

    if action == "status":
        pid = _read_pid()
        print("fan on" if pid else "fan off")
        return 0 

    if action == "on":
        if _read_pid():
            print("fan already on")
            return 0
        # Spawn detached holder
        try:
            subprocess.Popen([sys.executable, "-u", __file__, "_hold"],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             start_new_session=True)
            # Give it a moment to write the pid
            time.sleep(0.2)
            print("fan on")
            return 0
        except Exception as e:
            print(f"fan start failed: {e}")
            return 1

    if action == "off":
        pid = _read_pid()
        if not pid:
            print("fan already off")
            return 0
        try:
            os.kill(pid, signal.SIGTERM)
            # wait a moment for cleanup
            time.sleep(0.3)
            print("fan off")
            return 0
        except Exception as e:
            print(f"fan stop failed: {e}")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
