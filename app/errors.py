"""Error types and HTTP mapping.

Only these messages may reach a client. Internal detail (stack traces, provider
payloads, credentials) stays in the server log.
"""

from __future__ import annotations


class ServiceError(Exception):
    """Base class for errors that map to a documented HTTP status code."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code

    def to_payload(self) -> dict[str, object]:
        """Return a safe, secret-free JSON body."""
        return {"error": {"code": self.code, "message": self.message}}


class BadRequestError(ServiceError):
    """Malformed JSON or structurally invalid request."""

    status_code = 400
    code = "invalid_request"


class SemanticError(ServiceError):
    """Well-formed request that is semantically unusable."""

    status_code = 422
    code = "semantically_invalid_request"


class OptimizationError(ServiceError):
    """The optimizer could not produce a plan for this scenario."""

    status_code = 500
    code = "optimization_failed"


class LLMError(Exception):
    """Any provider-side failure. Internal only; never returned verbatim."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable
