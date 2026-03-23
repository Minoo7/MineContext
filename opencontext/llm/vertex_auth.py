"""Vertex AI helpers."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)

VERTEX_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_REFRESH_SKEW = timedelta(minutes=5)
_VERTEX_EMBEDDING_FALLBACK_LOCATION = "us-central1"


def is_vertex_openai_base_url(base_url: str | None) -> bool:
    """Return True when the URL looks like a Vertex OpenAI-compatible endpoint."""
    if not base_url:
        return False

    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path.lower()

    return "aiplatform.googleapis.com" in hostname and "/endpoints/openapi" in path


def normalize_vertex_model_name(model: str) -> str:
    """
    Vertex OpenAI endpoints expect publisher-qualified Gemini model names.

    Keep user input unchanged if it is already qualified or not a Gemini model.
    """
    normalized = (model or "").strip()
    if not normalized or "/" in normalized:
        return normalized
    if normalized.startswith("gemini"):
        return f"google/{normalized}"
    return normalized


def split_vertex_model_name(model: str) -> tuple[str, str]:
    """Split a Vertex model into publisher and model id."""
    normalized = (model or "").strip()
    if not normalized:
        raise ValueError("Vertex model must be provided")
    if "/" in normalized:
        publisher, model_id = normalized.split("/", 1)
        return publisher, model_id
    return "google", normalized


def build_vertex_predict_url(base_url: str, model: str) -> str:
    """Build the native Vertex publisher-model predict URL for embeddings."""
    parsed = urlparse(base_url)
    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 6 or path_parts[1] != "projects" or path_parts[3] != "locations":
        raise ValueError(f"Unsupported Vertex base URL: {base_url}")

    api_version = path_parts[0]
    project_id = path_parts[2]
    location = path_parts[4]
    publisher, model_id = split_vertex_model_name(model)

    hostname = parsed.netloc
    if location == "global":
        # Vertex text embedding REST calls are documented on regional endpoints.
        location = _VERTEX_EMBEDDING_FALLBACK_LOCATION
        hostname = f"{location}-aiplatform.googleapis.com"

    return (
        f"{parsed.scheme}://{hostname}/{api_version}/projects/{project_id}/locations/{location}/"
        f"publishers/{publisher}/models/{model_id}:predict"
    )


class VertexAccessTokenProvider:
    """Fetch and refresh Vertex access tokens via ADC, with a static fallback token."""

    def __init__(self, fallback_token: str | None = None):
        self._fallback_token = (fallback_token or "").strip() or None
        self._credentials = None
        self._request = None
        self._lock = threading.Lock()

    def get_access_token(self) -> str:
        """Return a valid access token, refreshing ADC credentials when available."""
        try:
            credentials = self._ensure_credentials()
        except Exception as exc:
            if self._fallback_token:
                logger.warning(
                    f"Falling back to the configured Vertex access token because ADC refresh failed: {exc}"
                )
                return self._fallback_token
            raise RuntimeError(
                "Vertex AI requires Google Cloud ADC credentials or a temporary access token."
            ) from exc

        if self._should_refresh(credentials):
            with self._lock:
                credentials = self._ensure_credentials()
                if self._should_refresh(credentials):
                    credentials.refresh(self._request)

        if getattr(credentials, "token", None):
            return credentials.token

        if self._fallback_token:
            return self._fallback_token

        raise RuntimeError("Failed to obtain a Vertex AI access token from ADC.")

    def _ensure_credentials(self):
        if self._credentials is None or self._request is None:
            import google.auth
            from google.auth.transport.requests import Request

            credentials, _project = google.auth.default(scopes=[VERTEX_SCOPE])
            self._credentials = credentials
            self._request = Request()
        return self._credentials

    @staticmethod
    def _should_refresh(credentials) -> bool:
        expiry = getattr(credentials, "expiry", None)
        if not getattr(credentials, "token", None):
            return True
        if expiry is None:
            return not getattr(credentials, "valid", False)
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return expiry <= datetime.now(timezone.utc) + _REFRESH_SKEW
