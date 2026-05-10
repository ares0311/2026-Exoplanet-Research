# Background Search Scheduler

## Purpose

The background search automation is designed to run one bounded invocation at a time:

```bash
exo background-run-once
```

Schedulers should call this command at a fixed cadence, capture output, and avoid overlapping runs. The command writes durable state to the top-level SQLite log database:

```text
logs/background_search.sqlite3
```

## Required Defaults

- Use fixture/local inputs by default.
- Do not enable live network access unless a future explicit configuration supports and logs it.
- Preserve stdout and stderr from scheduled runs.
- Enforce a runtime limit appropriate to the target pool.
- Avoid overlapping invocations. The command briefly waits for the background run lock before recording a blocked run.
- Notify the user only on failure or needs-follow-up events.
- Require explicit human approval before external submission or contact.

## Scheduler Exit Codes

By default, CLI commands emit JSON and return success when the command itself completed. Schedulers that need alert-friendly exit codes should pass:

```bash
exo background-run-once --scheduler-exit-codes
```

Exit code policy:

| Code | Meaning |
| --- | --- |
| `0` | Command completed with no scheduler-alert outcome. |
| `20` | A target needs follow-up. |
| `30` | The run was blocked, including lock wait timeout. |
| `40` | Configuration error. |
| `50` | Internal error. |

Schedulers can also consume:

```bash
exo scheduler-notification-summary
```

This command emits a compact JSON object with run id, target id, outcome, alert flag, reason, and report paths.

## Portable Cron Example

This example runs once per day at 03:15 local time. Adjust paths for the local checkout and Python environment.

```cron
15 3 * * * cd "/path/to/2026 Exoplanet Research" && exo background-run-once --config-path configs/background_search_v0.json --scheduler-exit-codes >> logs/background-search.cron.log 2>&1
```

## macOS launchd Example

Save a property list such as `com.exotoolkit.background-search.plist` under `~/Library/LaunchAgents/`, then load it with `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.exotoolkit.background-search.plist`.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.exotoolkit.background-search</string>

  <key>WorkingDirectory</key>
  <string>/path/to/2026 Exoplanet Research</string>

  <key>ProgramArguments</key>
  <array>
    <string>/path/to/exo</string>
    <string>background-run-once</string>
    <string>--config-path</string>
    <string>configs/background_search_v0.json</string>
    <string>--scheduler-exit-codes</string>
  </array>

  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>3</integer>
    <key>Minute</key>
    <integer>15</integer>
  </dict>

  <key>StandardOutPath</key>
  <string>/path/to/2026 Exoplanet Research/logs/background-search.launchd.out.log</string>

  <key>StandardErrorPath</key>
  <string>/path/to/2026 Exoplanet Research/logs/background-search.launchd.err.log</string>

  <key>RunAtLoad</key>
  <false/>
</dict>
</plist>
```

## systemd Timer Sketch

For Linux systems, pair a oneshot service with a timer. Keep the same one-run command and top-level SQLite log path.

```ini
[Service]
Type=oneshot
WorkingDirectory=/path/to/2026 Exoplanet Research
ExecStart=/path/to/exo background-run-once
```

```ini
[Timer]
OnCalendar=*-*-* 03:15:00
Persistent=true
```
