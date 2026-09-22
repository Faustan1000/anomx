"""Structured terminal failures from agent model backends."""

from __future__ import annotations


class AgentBackendError(RuntimeError):
    """A model request, tool loop, or context compression could not complete."""

    def __init__(self, message: str, *, code: str = "model_request_failed") -> None:
        super().__init__(message)
        self.code = code


class BackendFailure(str):
    """String-compatible error result for CLI callers of the runtime.

    Integrations must call ``raise_for_status`` before treating the result as
    an assistant answer or applying string operations that discard its type.
    """

    code: str

    def __new__(cls, message: str, *, code: str = "model_request_failed") -> BackendFailure:
        result = super().__new__(cls, message)
        result.code = code
        return result

    def raise_for_status(self) -> None:
        raise AgentBackendError(str(self), code=self.code)
