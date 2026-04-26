# Systemd Watchdog Implementation Plan

## Status
⏳ **Blocked on root access** — systemd service file requires sudo/root to edit

## Required Changes

### 1. Update `/etc/systemd/system/quant-bot.service`

**Change**: Replace deprecated `StartLimitIntervalSec=300` and add watchdog support.

```diff
--- a/quant-bot.service (current)
+++ b/quant-bot.service (target)
@@ -14,8 +14,9 @@
 ExecStart=/mnt/data/quant/.venv/bin/python3 -u src/scheduler/main.py
 Restart=on-failure
 RestartSec=30
-StartLimitBurst=3
-StartLimitIntervalSec=300
+StartLimitBurst=5
+StartLimitIntervalSec=3600
+WatchdogSec=300
 StandardOutput=journal
 StandardError=journal
 SyslogIdentifier=quant-bot
```

**Why**:
- Remove deprecated key `StartLimitIntervalSec` (fails in systemd v250+)
- Increase `StartLimitBurst` to 5 to match new interval (3600s = 1 hour)
- Add `WatchdogSec=300` to enable 5-minute watchdog timeout
- Watchdog requires application to send `sd_notify(0, "WATCHDOG=1")` periodically

### 2. Application-Side Watchdog Handler

Add to `/mnt/data/quant/src/scheduler/main.py`:

```python
import systemd.daemon
import signal

# Watchdog timeout in seconds (must match WatchdogSec, default 300)
WATCHDOG_TIMEOUT = 300

def setup_watchdog():
    """Initialize systemd watchdog."""
    if not systemd.daemon.booted():
        logger.info("Not running under systemd; watchdog disabled")
        return False
    
    usec = systemd.daemon.watchdog_enabled()
    if usec <= 0:
        logger.info("Systemd watchdog not enabled")
        return False
    
    logger.info(f"Systemd watchdog enabled; timeout={usec}usec")
    return True

def send_watchdog_ping():
    """Send heartbeat to systemd watchdog."""
    try:
        systemd.daemon.notify("WATCHDOG=1")
        # logger.debug("Watchdog ping sent")  # Too noisy; keep silent
    except Exception as e:
        logger.warning(f"Watchdog ping failed: {e}")

# In main scheduler loop, call send_watchdog_ping() every 60s
# (must be < WatchdogSec / 2 for safety)
```

### 3. Install systemd-python

```bash
pip install systemd-python
# or
apt-get install python3-systemd
```

## Testing Plan

### Pre-Change Validation
1. Check current systemd config:
   ```bash
   systemctl cat quant-bot.service | grep -E "StartLimit|Watchdog"
   ```
   Expected: Shows deprecated `StartLimitIntervalSec`

2. Check for warnings:
   ```bash
   journalctl -u quant-bot -p "err" -p "warning" --since "1 hour ago" | grep "StartLimit"
   ```
   Expected: Multiple warnings about unknown key

### Post-Change Validation
1. Reload systemd:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart quant-bot
   ```

2. Verify no warnings:
   ```bash
   journalctl -u quant-bot -p "err" -p "warning" --since "5 minutes ago"
   ```
   Expected: **Zero** warnings about StartLimit

3. Verify watchdog is active:
   ```bash
   systemctl show quant-bot.service -p WatchdogSec
   ```
   Expected: `WatchdogSec=5min` (300 seconds)

4. Verify application sends pings:
   ```bash
   journalctl -u quant-bot | grep -i "watchdog" | tail -5
   ```
   Expected: Watchdog ping messages every 60s

5. Verify uptime unchanged:
   ```bash
   systemctl status quant-bot.service | grep "Active:"
   ```
   Expected: Continuous uptime; no restarts after change

## Rollback Plan

If watchdog causes issues:
```bash
# Revert to previous service file
sudo git checkout /etc/systemd/system/quant-bot.service
sudo systemctl daemon-reload
sudo systemctl restart quant-bot
```

## Timeline
- **Apply systemd changes**: 10 min (requires root)
- **Deploy code changes**: 5 min (review + restart)
- **Validation**: 10 min
- **Total**: ~25 min downtime (restart only)

## Blocker
🔴 **Requires root/sudo access to edit `/etc/systemd/system/quant-bot.service`**

Unblock by: CEO or system administrator running the above `systemctl daemon-reload && restart` commands after code is ready.

## Follow-Up Work
Once watchdog is live:
1. Add health endpoint (HTTP :8000/health) for external monitoring
2. Add alert trigger if watchdog expires
3. Document in ops runbook
