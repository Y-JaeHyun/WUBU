"""Tests for health check HTTP endpoint."""

import unittest
from unittest.mock import MagicMock, patch

from src.scheduler.health_endpoint import HealthCheckHandler, HealthEndpointServer


class TestHealthEndpointServer(unittest.TestCase):
    """Test HealthEndpointServer lifecycle."""

    def test_server_initialization(self):
        """Should initialize with default parameters."""
        server = HealthEndpointServer()
        self.assertEqual(server.host, "127.0.0.1")
        self.assertEqual(server.port, 8000)
        self.assertIsNone(server.server)
        self.assertIsNone(server.thread)

    def test_server_custom_host_port(self):
        """Should accept custom host and port."""
        server = HealthEndpointServer(host="0.0.0.0", port=9000)
        self.assertEqual(server.host, "0.0.0.0")
        self.assertEqual(server.port, 9000)

    def test_server_is_running_initial(self):
        """Should report not running initially."""
        server = HealthEndpointServer()
        self.assertFalse(server.is_running())

    @patch("src.scheduler.health_endpoint.HTTPServer")
    @patch("threading.Thread")
    def test_server_start_creates_thread(self, mock_thread_cls, mock_http_server_cls):
        """Should create and start thread when starting."""
        mock_bot = MagicMock()
        server = HealthEndpointServer()

        mock_server_instance = MagicMock()
        mock_http_server_cls.return_value = mock_server_instance

        mock_thread_instance = MagicMock()
        mock_thread_cls.return_value = mock_thread_instance

        server.start(mock_bot)

        # Verify HTTPServer was created
        mock_http_server_cls.assert_called_once()
        # Verify thread was created and started
        mock_thread_cls.assert_called_once()
        mock_thread_instance.start.assert_called_once()

    @patch("src.scheduler.health_endpoint.HTTPServer")
    @patch("threading.Thread")
    def test_server_stop(self, mock_thread_cls, mock_http_server_cls):
        """Should stop server cleanly."""
        mock_bot = MagicMock()
        server = HealthEndpointServer()

        mock_server_instance = MagicMock()
        mock_http_server_cls.return_value = mock_server_instance

        mock_thread_instance = MagicMock()
        mock_thread_instance.is_alive.return_value = False
        mock_thread_cls.return_value = mock_thread_instance

        server.start(mock_bot)
        server.stop()

        # Verify thread was joined
        mock_thread_instance.join.assert_called_once()
        # Verify server is no longer set
        self.assertIsNone(server.server)
        self.assertIsNone(server.thread)


if __name__ == "__main__":
    unittest.main()
