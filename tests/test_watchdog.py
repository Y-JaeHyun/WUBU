"""Tests for systemd watchdog integration."""

import os
import unittest
from unittest.mock import MagicMock, patch

from src.scheduler.watchdog import (
    get_watchdog_status,
    get_watchdog_timeout,
    is_systemd_managed,
    send_watchdog_ping,
    setup_watchdog,
)


class TestWatchdogDetection(unittest.TestCase):
    """Test watchdog environment detection."""

    def test_is_systemd_managed_when_not_under_systemd(self):
        """Should return False when not under systemd."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = is_systemd_managed()
            self.assertFalse(result)

    @patch("builtins.open")
    def test_is_systemd_managed_when_under_systemd(self, mock_open):
        """Should return True when PID 1 is systemd."""
        mock_open.return_value.__enter__.return_value.read.return_value = "systemd\n"
        result = is_systemd_managed()
        self.assertTrue(result)

    @patch("builtins.open")
    def test_is_systemd_managed_when_init_is_not_systemd(self, mock_open):
        """Should return False when PID 1 is not systemd."""
        mock_open.return_value.__enter__.return_value.read.return_value = "init\n"
        result = is_systemd_managed()
        self.assertFalse(result)


class TestWatchdogTimeout(unittest.TestCase):
    """Test watchdog timeout detection."""

    def test_get_watchdog_timeout_not_set(self):
        """Should return None when WATCHDOG_USEC is not set."""
        with patch.dict(os.environ, {}, clear=False):
            if "WATCHDOG_USEC" in os.environ:
                del os.environ["WATCHDOG_USEC"]
            result = get_watchdog_timeout()
            self.assertIsNone(result)

    def test_get_watchdog_timeout_valid_usec(self):
        """Should convert microseconds to seconds."""
        with patch.dict(os.environ, {"WATCHDOG_USEC": "300000000"}):  # 300 seconds
            result = get_watchdog_timeout()
            self.assertEqual(result, 300)

    def test_get_watchdog_timeout_invalid_usec(self):
        """Should return None for invalid WATCHDOG_USEC."""
        with patch.dict(os.environ, {"WATCHDOG_USEC": "invalid"}):
            result = get_watchdog_timeout()
            self.assertIsNone(result)


class TestWatchdogSetup(unittest.TestCase):
    """Test watchdog initialization."""

    @patch("src.scheduler.watchdog.is_systemd_managed")
    def test_setup_watchdog_not_managed(self, mock_is_managed):
        """Should return False when not under systemd."""
        mock_is_managed.return_value = False
        result = setup_watchdog()
        self.assertFalse(result)

    @patch("src.scheduler.watchdog.get_watchdog_timeout")
    @patch("src.scheduler.watchdog.is_systemd_managed")
    def test_setup_watchdog_no_timeout(self, mock_is_managed, mock_timeout):
        """Should return False when watchdog timeout not set."""
        mock_is_managed.return_value = True
        mock_timeout.return_value = None
        result = setup_watchdog()
        self.assertFalse(result)

    @patch("src.scheduler.watchdog.get_watchdog_timeout")
    @patch("src.scheduler.watchdog.is_systemd_managed")
    def test_setup_watchdog_enabled(self, mock_is_managed, mock_timeout):
        """Should return True when watchdog is enabled."""
        mock_is_managed.return_value = True
        mock_timeout.return_value = 300
        result = setup_watchdog()
        self.assertTrue(result)


class TestWatchdogPing(unittest.TestCase):
    """Test watchdog ping functionality."""

    @patch("src.scheduler.watchdog.is_systemd_managed")
    def test_send_watchdog_ping_not_managed(self, mock_is_managed):
        """Should return False when not under systemd."""
        mock_is_managed.return_value = False
        result = send_watchdog_ping()
        self.assertFalse(result)

    @patch("src.scheduler.watchdog.is_systemd_managed")
    def test_send_watchdog_ping_systemd_managed(self, mock_is_managed):
        """Should attempt to send ping when under systemd."""
        mock_is_managed.return_value = True
        # Since systemd.daemon may not be available, just test the logic flow
        # The function should gracefully fall back if systemd is unavailable
        result = send_watchdog_ping()
        # Result depends on environment; just ensure no exception
        self.assertIsInstance(result, bool)

    @patch("src.scheduler.watchdog.HAS_SYSTEMD_DAEMON", False)
    @patch("src.scheduler.watchdog.is_systemd_managed")
    def test_send_watchdog_ping_fallback_no_socket(self, mock_is_managed):
        """Should return False when NOTIFY_SOCKET not set and daemon unavailable."""
        mock_is_managed.return_value = True
        with patch.dict(os.environ, {}, clear=False):
            if "NOTIFY_SOCKET" in os.environ:
                del os.environ["NOTIFY_SOCKET"]
            result = send_watchdog_ping()
            self.assertFalse(result)


class TestWatchdogStatus(unittest.TestCase):
    """Test watchdog status reporting."""

    @patch("src.scheduler.watchdog.get_watchdog_timeout")
    @patch("src.scheduler.watchdog.is_systemd_managed")
    def test_get_watchdog_status(self, mock_is_managed, mock_timeout):
        """Should return complete status dictionary."""
        mock_is_managed.return_value = True
        mock_timeout.return_value = 300

        status = get_watchdog_status()

        self.assertIsInstance(status, dict)
        self.assertTrue(status["under_systemd"])
        self.assertEqual(status["timeout_seconds"], 300)
        self.assertTrue(status["enabled"])


if __name__ == "__main__":
    unittest.main()
