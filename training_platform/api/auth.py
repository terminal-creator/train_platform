"""
Authentication Middleware for API Key validation
"""

import hashlib
import secrets
from typing import Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime
import logging

from ..core.database import engine, Session, ApiKeyRepository

logger = logging.getLogger(__name__)


# Public routes that don't require authentication
PUBLIC_ROUTES = {
    "/",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/v1",
}


# Routes that don't require API key (for development)
# In production, you should remove these or add proper auth
DEV_EXEMPTED_ROUTES = {
    "/api/v1/compute",  # Compute calculator (read-only)
}


def generate_api_key() -> str:
    """Generate a secure API key"""
    return secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    """Hash an API key for storage"""
    return hashlib.sha256(key.encode()).hexdigest()


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate API keys for protected routes.

    API keys should be provided in the X-API-KEY header.
    """

    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next):
        # Skip authentication if disabled (for development)
        if not self.enabled:
            return await call_next(request)

        # Check if route is public
        if self._is_public_route(request.url.path):
            return await call_next(request)

        # Check if route is exempted (for development)
        if self._is_exempted_route(request.url.path):
            logger.debug(f"Route {request.url.path} is exempted from auth (dev mode)")
            return await call_next(request)

        # Get API key from header
        api_key = request.headers.get("X-API-KEY")

        if not api_key:
            logger.warning(f"Missing API key for {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "detail": "API key required",
                    "error": "Missing X-API-KEY header"
                }
            )

        # Validate API key
        try:
            with Session(engine) as session:
                repo = ApiKeyRepository(session)

                # Hash the provided key for lookup
                key_hash = hash_api_key(api_key)
                db_key = repo.get_by_key(key_hash)

                if not db_key:
                    logger.warning(f"Invalid API key attempted for {request.url.path}")
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={
                            "detail": "Invalid API key",
                            "error": "API key not found"
                        }
                    )

                if not db_key.is_active:
                    logger.warning(f"Inactive API key attempted for {request.url.path}")
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={
                            "detail": "Invalid API key",
                            "error": "API key is inactive"
                        }
                    )

                # Check expiration
                if db_key.expires_at and db_key.expires_at < datetime.utcnow():
                    logger.warning(f"Expired API key attempted for {request.url.path}")
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={
                            "detail": "Invalid API key",
                            "error": "API key has expired"
                        }
                    )

                # Update last used timestamp
                repo.update_last_used(db_key)

                # Store key info in request state for later use
                request.state.api_key_name = db_key.name
                request.state.api_key_id = db_key.id

        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "detail": "Internal server error",
                    "error": "Authentication failed"
                }
            )

        # Proceed with request
        response = await call_next(request)
        return response

    def _is_public_route(self, path: str) -> bool:
        """Check if route is public"""
        return path in PUBLIC_ROUTES

    def _is_exempted_route(self, path: str) -> bool:
        """Check if route is exempted from auth (for development)"""
        for exempted in DEV_EXEMPTED_ROUTES:
            if path.startswith(exempted):
                return True
        return False


def create_default_api_key() -> Optional[str]:
    """
    Create a default API key for initial setup.
    Returns the plaintext key (only shown once).
    """
    try:
        with Session(engine) as session:
            repo = ApiKeyRepository(session)

            # Check if any keys exist
            existing_keys = repo.list_keys(active_only=False)
            if existing_keys:
                logger.info("API keys already exist, skipping default key creation")
                return None

            # Generate new key
            plaintext_key = generate_api_key()
            key_hash = hash_api_key(plaintext_key)

            # Create API key record
            from ..core.database import ApiKey
            api_key = ApiKey(
                key=key_hash,
                name="Default API Key",
                is_active=True,
                rate_limit_per_minute=1000,
            )

            repo.create(api_key)
            logger.info("Created default API key")

            return plaintext_key

    except Exception as e:
        logger.error(f"Failed to create default API key: {e}")
        return None
