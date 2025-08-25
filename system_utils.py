# voice_assistant/system_utils.py
import os
import re
import platform
import subprocess
import time
import shlex
from typing import Dict, Any, Optional

try:
    import psutil
except ImportError:
    psutil = None

from .config import config

_IPv4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_MAC_RE = re.compile(r"\b([0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}\b")

def run(cmd: str, timeout: int = 6) -> str:
    """Run a shell command and return output"""
    try:
        out = subprocess.check_output(
            cmd, shell=True, stderr=subprocess.STDOUT, 
            text=True, timeout=timeout
        )
        return out.strip()
    except Exception as e:
        return f"[cmd-failed] {cmd} :: {e}"

def redact(s: str) -> str:
    """Redact sensitive information from text"""
    s = _IPv4_RE.sub("<IP>", s or "")
    s = _MAC_RE.sub("<MAC>", s)
    return s

def clamp(s: str, n: int = None) -> str:
    """Truncate text with ellipsis in the middle"""
    if n is None:
        n = config.max_log_chars
        
    if not s:
        return ""
    if len(s) <= n:
        return s
        
    half = n // 2
    return s[:half] + "\n[...] (truncated)\n" + s[-half:]

def _pi_temp() -> str:
    """Get Raspberry Pi temperature"""
    out = run("vcgencmd measure_temp 2>/dev/null | cut -d= -f2")
    if out and "cmd-failed" not in out:
        return out
        
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            c = float(f.read().strip()) / 1000.0
        return f"{c:.1f}'C"
    except Exception:
        return "Unknown"

def _throttled() -> str:
    """Get throttling status"""
    out = run("vcgencmd get_throttled 2>/dev/null | awk -F= '{print $2}'")
    return out if out and "cmd-failed" not in out else "Unknown"

def _ip_addrs() -> str:
    """Get IP addresses"""
    out = run("ip -4 addr | awk '/inet /{print $2, $NF}'")
    return redact(out)

def _top_cpu() -> str:
    """Get top CPU processes"""
    return run("ps -eo pid,comm,%cpu,%mem --sort=-%cpu | head -n 8")

def collect_system_snapshot() -> Dict[str, Any]:
    """Collect comprehensive system information"""
    if psutil:
        try:
            load1, load5, load15 = os.getloadavg()
        except Exception:
            load1 = load5 = load15 = -1.0
            
        vm = psutil.virtual_memory()._asdict()
        du = psutil.disk_usage("/")._asdict()
        uptime = int(time.time() - psutil.boot_time())
        cpu_pct = psutil.cpu_percent(interval=0.2)
    else:
        try:
            load1, load5, load15 = os.getloadavg()
        except Exception:
            load1 = load5 = load15 = -1.0
            
        vm = {"total": 0, "available": 0, "percent": -1}
        du = {"total": 0, "used": 0, "percent": -1}
        uptime = int(run("awk '{print int($1)}' /proc/uptime") or "0")
        cpu_pct = -1

    return {
        "uname": platform.uname()._asdict(),
        "uptime_seconds": uptime,
        "loadavg": {"1m": load1, "5m": load5, "15m": load15},
        "cpu_percent": cpu_pct,
        "memory": {
            "total": vm.get("total"), 
            "available": vm.get("available"), 
            "percent": vm.get("percent")
        },
        "disk_root": {
            "total": du.get("total"), 
            "used": du.get("used"), 
            "percent": du.get("percent")
        },
        "pi_temp": _pi_temp(),
        "throttled_flags": _throttled(),
        "ip_addrs": _ip_addrs(),
        "top_cpu": _top_cpu(),
        "journal_errors": clamp(redact(run("journalctl -p 3 -xb --no-pager -n 50 2>/dev/null")), 1600),
        "dmesg_tail": clamp(redact(run("dmesg --ctime --color=never | tail -n 60")), 1600),
    }

def get_service_log(service: str, n: int = None) -> str:
    """Get service logs from journalctl"""
    if n is None:
        n = config.default_log_lines
        
    service = re.sub(r"[^a-zA-Z0-9_.@:-]", "", service)
    out = run(f"journalctl -u {service} --no-pager -n {n}")
    return clamp(redact(out))

def tail_file(path: str, n: int = None) -> str:
    """Tail a file with security restrictions"""
    if n is None:
        n = config.default_log_lines
        
    # Security: Only allow certain paths
    allowed_paths = ["/var/log/", "/tmp/"]
    if not any(path.startswith(allowed) for allowed in allowed_paths):
        return "Access denied. Only /var/log/* or /tmp/* are allowed."
        
    if not os.path.exists(path):
        return f"File not found: {path}"
        
    out = run(f"tail -n {n} {shlex.quote(path)}")
    return clamp(redact(out))

def ai_status_injection_needed(user_text: str) -> bool:
    """Check if system status should be injected into prompt"""
    if not user_text:
        return False
        
    return any(re.search(p, user_text, re.I) for p in config.status_patterns)