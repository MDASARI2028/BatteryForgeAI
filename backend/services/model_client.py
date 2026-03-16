"""
Model Client — Centralized RunPod API Client

Replaces all direct Google Gemini SDK calls with HTTP requests to a RunPod endpoint
serving open-source models (e.g. Qwen, Llama).

This is the ONLY module that communicates with RunPod.
All other services call model_client.generate() or model_client.generate_async().
"""

import os
import json
import time
import base64
import asyncio
import logging
from typing import Optional

import httpx

logger = logging.getLogger("model_client")


# ==========================================
# Error Classes
# ==========================================

class ModelClientError(Exception):
    """Base exception for ModelClient failures."""
    pass


class RunPodConnectionError(ModelClientError):
    """RunPod endpoint unreachable after all retries."""
    pass


class RunPodTimeoutError(ModelClientError):
    """Request timed out waiting for model inference."""
    pass


class RunPodAPIError(ModelClientError):
    """RunPod returned a non-200 HTTP status."""
    def __init__(self, status_code: int, detail: str = ""):
        self.status_code = status_code
        super().__init__(f"RunPod API error {status_code}: {detail}")


class RunPodResponseError(ModelClientError):
    """RunPod returned 200 but output is empty, unparseable, or malformed."""
    pass


# ==========================================
# Constants
# ==========================================

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)

MAX_IMAGE_SIZE_MB = 20
MAX_PROMPT_LENGTH = int(os.getenv("RUNPOD_MAX_PROMPT_CHARS", "60000"))


# ==========================================
# Model Client
# ==========================================

class ModelClient:
    """
    Centralized HTTP client for RunPod API.

    Supports:
      - Synchronous calls via generate()
      - Asynchronous calls via generate_async()
      - Text prompts and optional base64 image input
      - Retry with exponential backoff
      - Structured error classification
      - Connection pooling via httpx
    """

    def __init__(self):
        self.endpoint = os.getenv("RUNPOD_ENDPOINT", "")
        self.api_key = os.getenv("RUNPOD_API_KEY", "")

        if not self.endpoint.strip():
            raise ValueError("RUNPOD_ENDPOINT environment variable not set or is empty")
        if not self.api_key.strip():
            raise ValueError("RUNPOD_API_KEY environment variable not set or is empty")
        self.max_retries = int(os.getenv("RUNPOD_MAX_RETRIES", "2"))

        # Tiered timeout configuration
        self._timeout = httpx.Timeout(
            connect=float(os.getenv("RUNPOD_TIMEOUT_CONNECT", "10")),
            read=float(os.getenv("RUNPOD_TIMEOUT_READ", "120")),
            write=30.0,
            pool=10.0,
        )

        # Connection pool limits
        self._limits = httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
            keepalive_expiry=30,
        )

        # Lazy-initialized clients
        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------
    # Client Lifecycle
    # ------------------------------------------

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _get_sync_client(self) -> httpx.Client:
        """Lazy-init synchronous client with connection pooling."""
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(
                timeout=self._timeout,
                headers=self._headers(),
                limits=self._limits,
            )
        return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        """Lazy-init asynchronous client with connection pooling."""
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                timeout=self._timeout,
                headers=self._headers(),
                limits=self._limits,
            )
        return self._async_client

    async def close(self):
        """Shutdown hook — call on app shutdown to prevent socket leaks."""
        if self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()

    # ------------------------------------------
    # Input Validation
    # ------------------------------------------

    def _validate_inputs(self, prompt: str, image_b64: Optional[str] = None) -> Optional[str]:
        """
        Validate inputs before sending to RunPod.

        Returns:
            Error message string if invalid, None if OK.
        """
        if not prompt or not prompt.strip():
            return "Prompt is empty"

        if len(prompt) > MAX_PROMPT_LENGTH:
            # Don't reject — truncation happens in _build_payload.
            # Just log a warning here.
            logger.warning(
                "Prompt exceeds limit: %d chars (max %d). Will be truncated.",
                len(prompt), MAX_PROMPT_LENGTH
            )
            # Return None to allow the request to proceed with truncation
            return None

        if image_b64:
            # Quick pre-validation: ~1.33 bytes per base64 character.
            # 20MB * 1024 * 1024 * 1.33 ≈ 27.8 million characters
            if len(image_b64) > 30_000_000:
                return f"Image base64 string exceeds length limits (max ~{MAX_IMAGE_SIZE_MB}MB encoded)"
                
            # Validate base64 encoding (only for reasonably sized strings to avoid allocating too much memory)
            try:
                decoded = base64.b64decode(image_b64, validate=True)
            except Exception:
                return "Invalid base64 encoding in image data"

            # Check decoded size
            size_mb = len(decoded) / (1024 * 1024)
            if size_mb > MAX_IMAGE_SIZE_MB:
                return f"Image too large: {size_mb:.1f}MB (max {MAX_IMAGE_SIZE_MB}MB)"

        return None

    # ------------------------------------------
    # Payload Construction
    # ------------------------------------------

    def _build_payload(
        self,
        prompt: str,
        image_b64: Optional[str] = None,
        task: Optional[str] = None,
    ) -> dict:
        """
        Build the RunPod request payload.

        Truncates prompt if over MAX_PROMPT_LENGTH.
        """
        # Safe truncation
        if len(prompt) > MAX_PROMPT_LENGTH:
            prompt = prompt[:MAX_PROMPT_LENGTH] + "\n[TRUNCATED]"

        payload: dict = {"input": {"prompt": prompt}}

        if image_b64:
            payload["input"]["image"] = image_b64

        if task:
            payload["input"]["task"] = task

        return payload

    # ------------------------------------------
    # Response Extraction
    # ------------------------------------------

    def _extract_output(self, data: dict) -> str:
        """
        Normalize any RunPod response shape into a plain string.

        Handles:
          {"output": "text"}
          {"output": {"text": "..."}}
          {"output": {"generated_text": "..."}}
          {"output": ["text"]}
          {"output": [{"generated_text": "..."}]}
          {"text": "..."}
          {"result": "..."}

        Guarantees: Always returns str. Never returns None. Never raises.
        """
        output = data.get("output")

        # Fallback keys if "output" is missing at root
        if output is None and "data" in data and isinstance(data["data"], dict):
            inner = data["data"]
            if "output" in inner:
                output = inner["output"]

        if output is None:
            for key in ("result", "text", "generated_text", "response"):
                if key in data:
                    output = data[key]
                    break

        # Still nothing — return empty string (caller validates)
        if output is None:
            return ""

        if isinstance(output, str):
            return output

        if isinstance(output, dict):
            # OpenAI-compatible choices pattern
            if "choices" in output and isinstance(output["choices"], list):
                if output["choices"] and isinstance(output["choices"][0], dict):
                    first = output["choices"][0]
                    if "message" in first and isinstance(first["message"], dict) and "content" in first["message"]:
                        text = first["message"]["content"]
                        if text:
                            return text
                    if "text" in first:
                        text = first["text"]
                        if text:
                            return text

            # Try known text keys in priority order
            for key in ("text", "generated_text", "response", "content"):
                if key in output and isinstance(output[key], str):
                    return output[key]
            # Unknown dict structure — serialize it
            return json.dumps(output)

        if isinstance(output, list):
            # [{"generated_text": "..."}] pattern (HuggingFace-style)
            if output and isinstance(output[0], dict):
                first = output[0]
                for key in ("text", "generated_text", "response"):
                    if key in first:
                        return str(first[key])
            # Plain string list
            return "\n".join(str(item) for item in output)

        # Unexpected type — force to string
        return str(output)

    # ------------------------------------------
    # Synchronous Generate
    # ------------------------------------------

    def generate(
        self,
        prompt: str,
        image_b64: Optional[str] = None,
        task: Optional[str] = None,
    ) -> str:
        """
        Send a synchronous request to the RunPod endpoint.

        Args:
            prompt: The text prompt to send.
            image_b64: Optional base64-encoded image data.
            task: Optional task hint ("text" or "vision").

        Returns:
            The model's text response as a string.

        Raises:
            ValueError: If input validation fails.
            RunPodConnectionError: If the endpoint is unreachable.
            RunPodTimeoutError: If the request times out.
            RunPodAPIError: If RunPod returns a non-retryable HTTP error.
            RunPodResponseError: If the response is empty or unparseable.
        """
        # Validate
        error = self._validate_inputs(prompt, image_b64)
        if error:
            raise ValueError(f"ModelClient validation error: {error}")

        payload = self._build_payload(prompt, image_b64, task)
        client = self._get_sync_client()
        start = time.monotonic()
        
        prompt_len_before_truncation = len(prompt)
        prompt_len_after_truncation = len(payload["input"]["prompt"])

        logger.debug(
            "RunPod request | task=%s | pre_trunc_len=%d | post_trunc_len=%d | has_image=%s",
            task, prompt_len_before_truncation, prompt_len_after_truncation, image_b64 is not None,
        )


        for attempt in range(self.max_retries + 1):
            try:
                response = client.post(self.endpoint, json=payload)

                if response.status_code == 200:
                    # Guard: retry if response body is not valid JSON
                    try:
                        response_data = response.json()
                    except (json.JSONDecodeError, ValueError):
                        if attempt < self.max_retries:
                            logger.warning(
                                "RunPod returned non-JSON body, retrying | attempt=%d/%d",
                                attempt + 1, self.max_retries,
                            )
                            time.sleep(min(2 ** (attempt + 1), 8))
                            continue
                        raise RunPodResponseError(
                            "RunPod returned non-JSON response body"
                        )

                    output = self._extract_output(response_data)

                    # Validate non-empty output
                    if not output or not str(output).strip():
                        logger.error("RunPod returned empty output | task=%s", task)
                        raise RunPodResponseError(
                            "RunPod returned 200 but output is empty"
                        )
                        
                    if len(output) > 1_000_000:
                        logger.error("RunPod response too large (length: %d)", len(output))
                        raise RunPodResponseError("RunPod response too large (exceeds 1M characters)")

                    elapsed = (time.monotonic() - start) * 1000
                    logger.info(
                        "RunPod response | task=%s | status=200 | elapsed=%.0fms | output_len=%d",
                        task, elapsed, len(output),
                    )
                    return output

                # Retryable HTTP status
                if response.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    wait = min(2 ** (attempt + 1), 8)
                    logger.warning(
                        "RunPod retryable status %d | attempt=%d/%d | backoff=%.1fs",
                        response.status_code, attempt + 1, self.max_retries, wait,
                    )
                    time.sleep(wait)
                    continue

                # Non-retryable HTTP error
                raise RunPodAPIError(
                    response.status_code,
                    response.text[:500],
                )

            except RETRYABLE_EXCEPTIONS as e:
                if attempt < self.max_retries:
                    wait = min(2 ** (attempt + 1), 8)
                    logger.warning(
                        "RunPod connection/timeout error, retrying | attempt=%d/%d | backoff=%.1fs | error=%s",
                        attempt + 1, self.max_retries, wait, type(e).__name__,
                    )
                    time.sleep(wait)
                    continue

                # Final attempt exhausted
                elapsed = (time.monotonic() - start) * 1000
                logger.error(
                    "RunPod request failed after %d retries | elapsed=%.0fms | error=%s",
                    self.max_retries, elapsed, str(e),
                )
                if isinstance(e, httpx.TimeoutException):
                    raise RunPodTimeoutError(
                        f"RunPod timed out after {self.max_retries + 1} attempts: {e}"
                    ) from e
                raise RunPodConnectionError(
                    f"Cannot reach RunPod after {self.max_retries + 1} attempts: {e}"
                ) from e

        # Should not reach here, but safety net
        raise RunPodConnectionError("RunPod request failed: exhausted all retries")

    # ------------------------------------------
    # Asynchronous Generate
    # ------------------------------------------

    async def generate_async(
        self,
        prompt: str,
        image_b64: Optional[str] = None,
        task: Optional[str] = None,
    ) -> str:
        """
        Send an asynchronous request to the RunPod endpoint.

        Args:
            prompt: The text prompt to send.
            image_b64: Optional base64-encoded image data.
            task: Optional task hint ("text" or "vision").

        Returns:
            The model's text response as a string.

        Raises:
            ValueError: If input validation fails.
            RunPodConnectionError: If the endpoint is unreachable.
            RunPodTimeoutError: If the request times out.
            RunPodAPIError: If RunPod returns a non-retryable HTTP error.
            RunPodResponseError: If the response is empty or unparseable.
        """
        # Validate
        error = self._validate_inputs(prompt, image_b64)
        if error:
            raise ValueError(f"ModelClient validation error: {error}")

        payload = self._build_payload(prompt, image_b64, task)
        client = self._get_async_client()
        start = time.monotonic()

        prompt_len_before_truncation = len(prompt)
        prompt_len_after_truncation = len(payload["input"]["prompt"])
        
        logger.debug(
            "RunPod async request | task=%s | pre_trunc_len=%d | post_trunc_len=%d | has_image=%s",
            task, prompt_len_before_truncation, prompt_len_after_truncation, image_b64 is not None,
        )


        for attempt in range(self.max_retries + 1):
            try:
                response = await client.post(self.endpoint, json=payload)

                if response.status_code == 200:
                    # Guard: retry if response body is not valid JSON
                    try:
                        response_data = response.json()
                    except (json.JSONDecodeError, ValueError):
                        if attempt < self.max_retries:
                            logger.warning(
                                "RunPod returned non-JSON body, retrying | attempt=%d/%d",
                                attempt + 1, self.max_retries,
                            )
                            await asyncio.sleep(min(2 ** (attempt + 1), 8))
                            continue
                        raise RunPodResponseError(
                            "RunPod returned non-JSON response body"
                        )

                    output = self._extract_output(response_data)

                    # Validate non-empty output
                    if not output or not str(output).strip():
                        logger.error("RunPod returned empty output | task=%s", task)
                        raise RunPodResponseError(
                            "RunPod returned 200 but output is empty"
                        )
                        
                    if len(output) > 1_000_000:
                        logger.error("RunPod response too large (length: %d)", len(output))
                        raise RunPodResponseError("RunPod response too large (exceeds 1M characters)")

                    elapsed = (time.monotonic() - start) * 1000
                    logger.info(
                        "RunPod response | task=%s | status=200 | elapsed=%.0fms | output_len=%d",
                        task, elapsed, len(output),
                    )
                    return output

                # Retryable HTTP status
                if response.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    wait = min(2 ** (attempt + 1), 8)
                    logger.warning(
                        "RunPod retryable status %d | attempt=%d/%d | backoff=%.1fs",
                        response.status_code, attempt + 1, self.max_retries, wait,
                    )
                    await asyncio.sleep(wait)
                    continue

                # Non-retryable HTTP error
                raise RunPodAPIError(
                    response.status_code,
                    response.text[:500],
                )

            except RETRYABLE_EXCEPTIONS as e:
                if attempt < self.max_retries:
                    wait = min(2 ** (attempt + 1), 8)
                    logger.warning(
                        "RunPod connection/timeout error, retrying | attempt=%d/%d | backoff=%.1fs | error=%s",
                        attempt + 1, self.max_retries, wait, type(e).__name__,
                    )
                    await asyncio.sleep(wait)
                    continue

                # Final attempt exhausted
                elapsed = (time.monotonic() - start) * 1000
                logger.error(
                    "RunPod request failed after %d retries | elapsed=%.0fms | error=%s",
                    self.max_retries, elapsed, str(e),
                )
                if isinstance(e, httpx.TimeoutException):
                    raise RunPodTimeoutError(
                        f"RunPod timed out after {self.max_retries + 1} attempts: {e}"
                    ) from e
                raise RunPodConnectionError(
                    f"Cannot reach RunPod after {self.max_retries + 1} attempts: {e}"
                ) from e

        # Should not reach here, but safety net
        raise RunPodConnectionError("RunPod request failed: exhausted all retries")


# ==========================================
# Singleton Instance
# ==========================================

model_client = None
def get_model_client() -> ModelClient:
    global model_client
    if model_client is None:
        model_client = ModelClient()
    return model_client