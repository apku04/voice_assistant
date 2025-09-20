#!/usr/bin/env python3
from __future__ import annotations

import sys
import math

def read_sht31d() -> str:
    try:
        import board
        import adafruit_sht31d
    except Exception as e:
        return "sensor lib not available"

    try:
        i2c = board.I2C()
        sensor = adafruit_sht31d.SHT31D(i2c, address=0x44)
        sensor.heater = False
        t = sensor.temperature
        # Round up to the next whole degree (ceil)
        t_up = math.ceil(t)
        return f"System tempreture is {t_up} ° Celsius"
    except Exception as e:
        return f"sensor read failed: {e}"


def main():
    print(read_sht31d())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

