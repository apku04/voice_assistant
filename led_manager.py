# voice_assistant/led_manager.py
from __future__ import annotations

from typing import Dict, Tuple, Optional, Any

try:
    from gpiozero import Device, RGBLED
    from gpiozero.pins.lgpio import LGPIOFactory
    Device.pin_factory = LGPIOFactory()
except ImportError:
    RGBLED = None

from .config import AppConfig, get_config

class LEDManager:
    """Manager for RGB LED feedback"""
    
    def __init__(self, config: AppConfig | None = None):
        self.led = None
        self._config = config or get_config()
        self.colors = {
            "idle": self._config.idle_color,
            "think": self._config.think_color,
            "speak": self._config.speak_color,
            "error": self._config.error_color
        }
        
        self._initialize_led()
        
    def _initialize_led(self):
        """Initialize the RGB LED if available"""
        if RGBLED is None:
            print("[warn] RGB LED not available; continuing without LED.")
            return
            
        try:
            self.led = RGBLED(
                red=self._config.red_pin, 
                green=self._config.green_pin, 
                blue=self._config.blue_pin, 
                active_high=self._config.led_active_high, 
                pwm=True
            )
            self.set_color("idle")
        except Exception as e:
            print(f"[warn] RGB LED not active ({e}); continuing without LED.")
            self.led = None
            
    def set_color(self, color_name: str):
        """Set LED to a named color"""
        if not self.led:
            return
            
        color = self.colors.get(color_name)
        if color is not None:
            try:
                self.led.color = color
            except Exception as e:
                print(f"[warn] Failed to set LED color: {e}")
                
    def pulse(self, color_name: str = "think", period: float = 1.6):
        """Pulse the LED (to be called in a loop)"""
        if not self.led:
            return
            
        color = self.colors.get(color_name)
        if color is None:
            return
            
        try:
            import math
            import time
            
            phase = time.time() % period
            amp = 0.15 + 0.85 * (0.5 * (1 - math.cos(2 * math.pi * phase / period)))
            r, g, b = color
            self.led.color = (r * amp, g * amp, b * amp)
        except Exception as e:
            print(f"[warn] Failed to pulse LED: {e}")
