"""
Enhanced error handling for the API

Provides:
- Consistent error response format
- User-friendly error messages
- Detailed error information for debugging
- Error categorization
"""

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from typing import Any, Dict, Optional
import logging
import traceback

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base class for API errors"""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)


class NotFoundError(APIError):
    """Resource not found"""

    def __init__(self, resource: str, identifier: Any):
        super().__init__(
            message=f"{resource} 未找到: {identifier}",
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="NOT_FOUND",
            details={"resource": resource, "identifier": str(identifier)},
        )


class ValidationError(APIError):
    """Validation error"""

    def __init__(self, message: str, field: Optional[str] = None):
        details = {"field": field} if field else {}
        super().__init__(
            message=f"验证错误: {message}",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            details=details,
        )


class ConflictError(APIError):
    """Resource conflict"""

    def __init__(self, message: str):
        super().__init__(
            message=f"冲突: {message}",
            status_code=status.HTTP_409_CONFLICT,
            error_code="CONFLICT",
        )


class BadRequestError(APIError):
    """Bad request"""

    def __init__(self, message: str):
        super().__init__(
            message=f"请求错误: {message}",
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="BAD_REQUEST",
        )


class UnauthorizedError(APIError):
    """Unauthorized access"""

    def __init__(self, message: str = "未授权"):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="UNAUTHORIZED",
        )


class ForbiddenError(APIError):
    """Forbidden access"""

    def __init__(self, message: str = "禁止访问"):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="FORBIDDEN",
        )


class InternalServerError(APIError):
    """Internal server error"""

    def __init__(self, message: str = "服务器内部错误"):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="INTERNAL_SERVER_ERROR",
        )


def create_error_response(
    status_code: int,
    message: str,
    error_code: str = "ERROR",
    details: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    """Create a standardized error response"""
    content = {
        "error": {
            "code": error_code,
            "message": message,
            "details": details or {},
        }
    }
    return JSONResponse(status_code=status_code, content=content)


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle custom API errors"""
    logger.warning(
        f"API Error: {exc.error_code} - {exc.message}",
        extra={"details": exc.details},
    )
    return create_error_response(
        status_code=exc.status_code,
        message=exc.message,
        error_code=exc.error_code,
        details=exc.details,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions"""
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return create_error_response(
        status_code=exc.status_code,
        message=str(exc.detail),
        error_code="HTTP_ERROR",
    )


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors"""
    errors = exc.errors()
    logger.warning(f"Validation Error: {errors}")

    # Format validation errors for better user experience
    formatted_errors = []
    for error in errors:
        field = " -> ".join(str(x) for x in error["loc"][1:])  # Skip 'body'
        formatted_errors.append(
            {
                "field": field,
                "message": error["msg"],
                "type": error["type"],
            }
        )

    return create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        message="请求参数验证失败",
        error_code="VALIDATION_ERROR",
        details={"errors": formatted_errors},
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions"""
    logger.error(
        f"Unexpected error: {type(exc).__name__}: {str(exc)}",
        exc_info=True,
    )

    # Don't expose internal error details in production
    message = "服务器内部错误，请稍后重试"
    details = {}

    # In development, provide more details
    import os
    if os.getenv("ENV", "production") == "development":
        details = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        message=message,
        error_code="INTERNAL_SERVER_ERROR",
        details=details,
    )


def register_error_handlers(app):
    """Register all error handlers with the FastAPI app"""
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, general_exception_handler)
