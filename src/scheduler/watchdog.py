"""Systemd watchdog integration for quant trading bot.

Sends periodic heartbeats to systemd watchdog to ensure the bot is alive.
If systemd is not available, logs a warning and continues normally.
"""

import os
import socket
import subprocess
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Try to import systemd.daemon; if unavailable, we'll use subprocess fallback
try:
    import systemd.daemon
    HAS_SYSTEMD_DAEMON = True
except ImportError:
    HAS_SYSTEMD_DAEMON = False


def is_systemd_managed() -> bool:
    """Check if this process is running under systemd."""
    # Check if parent process is systemd (PID 1)
    try:
        with open("/proc/1/comm", "r") as f:
            init = f.read().strip()
            return init == "systemd"
    except (OSError, FileNotFoundError):
        return False


def get_watchdog_timeout() -> Optional[int]:
    """Get watchdog timeout in seconds from systemd environment variable.

    Systemd passes WATCHDOG_USEC in microseconds; we return seconds.
    """
    watchdog_usec = os.environ.get("WATCHDOG_USEC")
    if not watchdog_usec:
        return None

    try:
        usec = int(watchdog_usec)
        return usec // 1_000_000  # Convert microseconds to seconds
    except (ValueError, ZeroDivisionError):
        return None


def setup_watchdog() -> bool:
    """Initialize systemd watchdog if running under systemd.

    Returns:
        True if watchdog is enabled and ready, False otherwise.
    """
    if not is_systemd_managed():
        logger.info("Not running under systemd; watchdog disabled")
        return False

    timeout = get_watchdog_timeout()
    if not timeout:
        logger.warning("Systemd watchdog not enabled (WATCHDOG_USEC not set)")
        return False

    logger.info("Systemd watchdog enabled; timeout=%d seconds", timeout)
    return True


def send_watchdog_ping() -> bool:
    """Send heartbeat to systemd watchdog.

    Returns:
        True if ping was sent successfully, False otherwise.
    """
    if not is_systemd_managed():
        return False

    # Try native systemd.daemon first
    if HAS_SYSTEMD_DAEMON:
        try:
            systemd.daemon.notify("WATCHDOG=1")
            return True
        except Exception as e:
            logger.warning("Watchdog ping via systemd.daemon failed: %s", e)

    # Fallback: use systemctl notify via D-Bus or socket
    try:
        # Get the socket path from NOTIFY_SOCKET environment variable
        notify_socket = os.environ.get("NOTIFY_SOCKET")
        if not notify_socket:
            logger.debug("NOTIFY_SOCKET not set; watchdog unavailable")
            return False

        # If abstract socket (starts with @), convert to /run/systemd/notify
        if notify_socket.startswith("@"):
            notify_socket = "\0" + notify_socket[1:]

        # Send notification via Unix domain socket
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.sendto(b"WATCHDOG=1", notify_socket)
        sock.close()
        return True
    except Exception as e:
        logger.warning("Watchdog ping via Unix socket failed: %s", e)

    # Last resort: use systemctl
    try:
        pid = os.getpid()
        subprocess.run(
            ["systemctl", "notify", "--pid", str(pid), "WATCHDOG=1"],
            check=False,
            capture_output=True,
            timeout=2
        )
        return True
    except Exception as e:
        logger.debug("Watchdog ping via systemctl failed: %s", e)

    return False


def get_watchdog_status() -> dict:
    """Get current watchdog status.

    Returns:
        Dictionary with watchdog status information.
    """
    return {
        "under_systemd": is_systemd_managed(),
        "timeout_seconds": get_watchdog_timeout(),
        "has_systemd_daemon": HAS_SYSTEMD_DAEMON,
        "enabled": is_systemd_managed() and get_watchdog_timeout() is not None,
    }
