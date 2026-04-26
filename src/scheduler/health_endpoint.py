"""Health check HTTP endpoint for systemd and external monitoring.

Runs a simple HTTP server on port 8000 that responds to /health requests.
Designed to be lightweight and independent of main trading logic.
"""

import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any, Optional

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.scheduler.main import TradingBot

logger = get_logger(__name__)


class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health checks."""

    # Class variable to store reference to TradingBot
    trading_bot: Optional["TradingBot"] = None

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/ready":
            self._handle_ready()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "Not Found"}')

    def _handle_health(self) -> None:
        """Handle /health endpoint - basic liveness probe."""
        try:
            status = {
                "status": "alive",
                "timestamp": datetime.now().isoformat(),
                "service": "quant-bot",
            }

            # If trading bot is available, include additional status
            if self.trading_bot:
                try:
                    mdd = self.trading_bot.portfolio_tracker.get_mdd()
                    status["mdd"] = float(mdd)
                except Exception as e:
                    status["mdd_error"] = str(e)

                try:
                    is_trading = self.trading_bot.holidays.is_trading_day(
                        datetime.now(self.trading_bot.kis_client._tz).date()
                    )
                    status["trading_day"] = is_trading
                except Exception as e:
                    status["trading_day_error"] = str(e)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(status).encode())

        except Exception as e:
            logger.error("Health check handler error: %s", e)
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _handle_ready(self) -> None:
        """Handle /ready endpoint - readiness probe.

        Returns 200 if bot is ready to trade, 503 otherwise.
        """
        try:
            if not self.trading_bot:
                status_code = 503
                status = "initializing"
            else:
                # Check if critical systems are operational
                status_code = 200
                status = "ready"

                try:
                    # Check KIS client
                    if not self.trading_bot.kis_client.is_configured():
                        status_code = 503
                        status = "kis_not_configured"
                except Exception:
                    status_code = 503
                    status = "kis_check_failed"

            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": status}).encode())

        except Exception as e:
            logger.error("Ready check handler error: %s", e)
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default HTTP logging; use logger instead."""
        # Only log errors and unusual cases
        if "GET" in format or "POST" in format:
            logger.debug("Health endpoint: %s", format % args)


class HealthEndpointServer:
    """Health endpoint HTTP server running in a background thread."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        """Initialize health endpoint server.

        Args:
            host: Host to bind to (default: 127.0.0.1 for localhost only)
            port: Port to listen on (default: 8000)
        """
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self, trading_bot: "TradingBot") -> None:
        """Start the health endpoint server.

        Args:
            trading_bot: Reference to TradingBot instance for status checks
        """
        if self.server is not None:
            logger.warning("Health endpoint server already running")
            return

        try:
            # Store trading bot reference for handlers
            HealthCheckHandler.trading_bot = trading_bot

            # Create HTTP server
            self.server = HTTPServer((self.host, self.port), HealthCheckHandler)

            # Start in background thread
            self.thread = threading.Thread(
                target=self._run_server,
                daemon=True,
                name="HealthEndpoint"
            )
            self.thread.start()

            logger.info(
                "Health endpoint server started on http://%s:%d/health",
                self.host,
                self.port
            )

        except Exception as e:
            logger.error("Failed to start health endpoint server: %s", e)
            self.server = None

    def _run_server(self) -> None:
        """Run the HTTP server until stopped."""
        try:
            # Set socket timeout for graceful shutdown
            self.server.timeout = 1.0
            while not self._stop_event.is_set():
                self.server.handle_request()
        except Exception as e:
            logger.error("Health endpoint server error: %s", e)
        finally:
            if self.server:
                self.server.server_close()

    def stop(self) -> None:
        """Stop the health endpoint server."""
        if self.server is None:
            return

        self._stop_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                logger.warning("Health endpoint thread did not stop cleanly")

        self.server = None
        self.thread = None
        logger.info("Health endpoint server stopped")

    def is_running(self) -> bool:
        """Check if server is running."""
        return self.server is not None and self.thread is not None and self.thread.is_alive()
