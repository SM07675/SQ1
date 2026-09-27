import subprocess

KEY_PATH = r"C:\Users\sarve\key"
NODE_IP = "151.185.58.96"

script_content = r'''#!/usr/bin/env bash
# ==============================================================================
# SatQuery AI - Auto-Shutdown Idle Watchdog
# Automatically powers off VM if idle for 30 minutes to preserve cloud credits.
# ==============================================================================

IDLE_TIMEOUT_MINUTES=30
LOG_FILE="/var/log/satquery_watchdog.log"
DISABLE_FLAG="/opt/satquery/DISABLE_AUTO_SHUTDOWN"
LAST_ACTIVE_FILE="/tmp/satquery_last_activity"

# 1. Manual override flag check
if [ -f "$DISABLE_FLAG" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - [WATCHDOG] Disabled by flag file ($DISABLE_FLAG)" >> "$LOG_FILE"
    exit 0
fi

# 2. Check for active SSH administrative sessions
SSH_USERS=$(who | wc -l)
if [ "$SSH_USERS" -gt 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - [WATCHDOG] Active SSH session ($SSH_USERS users). Keeping alive." >> "$LOG_FILE"
    date +%s > "$LAST_ACTIVE_FILE"
    exit 0
fi

# 3. Check for API activity in satquery container (excluding passive healthchecks)
RECENT_REQUESTS=$(docker logs --since "${IDLE_TIMEOUT_MINUTES}m" satquery-backend 2>/dev/null | grep -E 'HTTP/1.[01]" [1-5][0-9][0-9]' | grep -v 'GET /health' | wc -l)
if [ "$RECENT_REQUESTS" -gt 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - [WATCHDOG] API requests detected ($RECENT_REQUESTS in last ${IDLE_TIMEOUT_MINUTES}m). Keeping alive." >> "$LOG_FILE"
    date +%s > "$LAST_ACTIVE_FILE"
    exit 0
fi

# 4. Grace period: Do not shut down within 30 minutes of system boot
UPTIME_MINS=$(awk '{print int($1/60)}' /proc/uptime)
if [ "$UPTIME_MINS" -lt "$IDLE_TIMEOUT_MINUTES" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - [WATCHDOG] Boot grace period active (uptime: ${UPTIME_MINS}m / ${IDLE_TIMEOUT_MINUTES}m)." >> "$LOG_FILE"
    exit 0
fi

# 5. Evaluate idle duration
if [ ! -f "$LAST_ACTIVE_FILE" ]; then
    date +%s > "$LAST_ACTIVE_FILE"
    exit 0
fi

LAST_ACTIVE=$(cat "$LAST_ACTIVE_FILE")
NOW=$(date +%s)
IDLE_SECS=$((NOW - LAST_ACTIVE))
IDLE_MINS=$((IDLE_SECS / 60))

if [ "$IDLE_MINS" -ge "$IDLE_TIMEOUT_MINUTES" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - [WATCHDOG] Server idle for ${IDLE_MINS} minutes (0 requests, 0 SSH). Shutting down to save credits..." >> "$LOG_FILE"
    cd /opt/satquery && docker compose stop
    /sbin/shutdown -h now
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') - [WATCHDOG] Idle for ${IDLE_MINS} / ${IDLE_TIMEOUT_MINUTES} minutes. Running normally." >> "$LOG_FILE"
fi
'''

# 1. Write the script on the remote server
upload_cmd = f"cat << 'EOF' > /opt/satquery/auto_shutdown.sh\n{script_content}\nEOF\nchmod +x /opt/satquery/auto_shutdown.sh"
subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-i", KEY_PATH, f"root@{NODE_IP}", upload_cmd], check=True)

# 2. Add cron job to run every 5 minutes
cron_setup = (
    "(crontab -l 2>/dev/null | grep -v 'auto_shutdown.sh'; "
    "echo '*/5 * * * * /opt/satquery/auto_shutdown.sh') | crontab -"
)
subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-i", KEY_PATH, f"root@{NODE_IP}", cron_setup], check=True)

# 3. Test execution of the script
test_run = "/opt/satquery/auto_shutdown.sh && tail -n 5 /var/log/satquery_watchdog.log"
res = subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-i", KEY_PATH, f"root@{NODE_IP}", test_run], capture_output=True, text=True, check=True)

print("CRON & SCRIPT INSTALLED SUCCESSFULLY!")
print("Initial Watchdog Test Output:")
print(res.stdout)
