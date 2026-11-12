from __future__ import annotations


LIVE_CONTROL_KEYS = {"live_control", "send_command", "execute_scan", "move_motor", "hardware"}


def validate_no_live_control_arguments(arguments: dict) -> str | None:
    lowered = {str(key).lower() for key in arguments.keys()}
    if lowered & LIVE_CONTROL_KEYS:
        return "live control arguments are blocked in the MVP"
    mode = str(arguments.get("mode", "")).lower()
    if mode in {"live", "hardware", "instrument"}:
        return "live instrument modes are blocked in the MVP"
    return None

