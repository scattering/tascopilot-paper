class TasCopilotError(Exception):
    """Base exception for co-pilot errors."""


class PermissionBlockedError(TasCopilotError):
    """Raised when a tool call is blocked by policy."""


class ToolUnavailableError(TasCopilotError):
    """Raised when a tool is registered but unavailable."""

