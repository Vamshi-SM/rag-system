"""Centralized exception types and handlers for the FastAPI app.

Every handler returns the same structured JSON shape:

    {"status": "error", "message": "..."}

so API clients only need to handle one error format regardless of
which layer of the application raised.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.utils.logger import get_logger

logger = get_logger(__name__)


class APIError(Exception):
    """Base class for all deliberately-raised API errors.

    Args:
        message: Human-readable message returned to the client.
        status_code: HTTP status code to respond with.
    """

    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class FolderNotFoundError(APIError):
    """Raised when ``POST /ingest`` is given a folder that doesn't exist."""

    def __init__(self, folder: str) -> None:
        super().__init__(f"Folder not found: {folder}", status.HTTP_404_NOT_FOUND)


class NoDocumentsFoundError(APIError):
    """Raised when a folder exists but contains no supported documents."""

    def __init__(self, folder: str) -> None:
        super().__init__(
            f"No supported documents found in folder: {folder}", status.HTTP_404_NOT_FOUND
        )


class PathNotAllowedError(APIError):
    """Raised when a requested ingest folder falls outside the allowed root.

    Prevents a caller from pointing ``POST /ingest`` at arbitrary
    filesystem locations (e.g. ``/etc``, ``~/.ssh``) via path traversal
    or an absolute path outside the configured data directory tree.
    """

    def __init__(self, folder: str, allowed_root: str) -> None:
        super().__init__(
            f"Folder '{folder}' is outside the allowed ingest root '{allowed_root}'.",
            status.HTTP_403_FORBIDDEN,
        )


class ServiceUnavailableError(APIError):
    """Raised when an upstream dependency (embedding API, LLM API) is down."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status.HTTP_503_SERVICE_UNAVAILABLE)


def _error_response(message: str, status_code: int) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"status": "error", "message": message})


def register_exception_handlers(app: FastAPI) -> None:
    """Register all centralized exception handlers on the FastAPI app."""

    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, exc: APIError) -> JSONResponse:
        logger.warning("API error on %s %s: %s", request.method, request.url.path, exc.message)
        return _error_response(exc.message, exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        first_error = exc.errors()[0] if exc.errors() else {}
        field_path = ".".join(str(part) for part in first_error.get("loc", []) if part != "body")
        message = f"Invalid request: {first_error.get('msg', 'validation failed')}"
        if field_path:
            message = f"Invalid request field '{field_path}': {first_error.get('msg', '')}"
        logger.warning(
            "Validation error on %s %s: %s", request.method, request.url.path, exc.errors()
        )
        return _error_response(message, status.HTTP_422_UNPROCESSABLE_CONTENT)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled error on %s %s: %s", request.method, request.url.path, exc
        )
        return _error_response(
            "An unexpected error occurred. Please try again later.",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
