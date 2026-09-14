"""
User-facing errors with plain-language messages.

Catch ``EvrpUserError`` (or ``ValueError``) in scripts and print ``str(exc)``.
"""


class EvrpUserError(ValueError):
    """
    Invalid usage order, missing prerequisites, or inconsistent inputs.

    Examples: selecting stations that need synthetic fillers before any customers exist;
    snapping a point farther than ``max_dist_m`` from the road network.
    """

    pass


class EvrpValidationError(EvrpUserError):
    """Generated instance failed structural / service-graph acceptance checks."""

    pass


def format_exception_for_user(exc: BaseException) -> str:
    """
    Single string suitable for logging or UI (no traceback).

    Uses the exception's message; for ``EvrpUserError`` you can mention the type in a header.
    """
    name = type(exc).__name__
    msg = str(exc).strip() or "(no message)"
    if isinstance(exc, EvrpUserError):
        return f"[{name}] {msg}"
    return f"[{name}] {msg}"
