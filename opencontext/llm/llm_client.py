# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
OpenContext module: llm_client
"""

import asyncio
import json
from urllib import error as urllib_error
from urllib import request as urllib_request

from enum import Enum
from typing import Any, Dict, List

from openai import APIError, AsyncOpenAI, OpenAI
from volcenginesdkarkruntime import Ark

from opencontext.llm.vertex_auth import (
    VertexAccessTokenProvider,
    build_vertex_predict_url,
    is_vertex_openai_base_url,
    normalize_vertex_model_name,
)
from opencontext.models.context import Vectorize
from opencontext.monitoring import record_processing_stage
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)
_VERTEX_ADC_PLACEHOLDERS = {"vertex-adc", "adc", "use-adc"}


class LLMProvider(Enum):
    OPENAI = "openai"
    DOUBAO = "doubao"
    VERTEX = "vertex"


class LLMType(Enum):
    CHAT = "chat"
    EMBEDDING = "embedding"


class LLMClient:
    def __init__(self, llm_type: LLMType, config: Dict[str, Any]):
        self.llm_type = llm_type
        self.config = config
        raw_model = config.get("model")
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        self.timeout = config.get("timeout", 300)
        self.provider = config.get("provider", LLMProvider.OPENAI.value)
        self.is_vertex_openai = self.provider == LLMProvider.VERTEX.value or is_vertex_openai_base_url(
            self.base_url
        )
        self.model = normalize_vertex_model_name(raw_model) if self.is_vertex_openai else raw_model
        if not self.base_url or not self.model:
            raise ValueError("Base URL and model must be provided")
        if not self.api_key and not self.is_vertex_openai:
            raise ValueError("API key, base URL, and model must be provided")
        self._vertex_token_provider = None
        if self.is_vertex_openai and self._should_use_vertex_adc():
            self._vertex_token_provider = VertexAccessTokenProvider()
        self._active_api_key = None
        self.client = None
        self.async_client = None
        self._refresh_clients(force=True)
        if self.provider == LLMProvider.DOUBAO.value and self.llm_type == LLMType.EMBEDDING:
            self.client = Ark(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
            self.async_client = None

    def _should_use_vertex_adc(self) -> bool:
        token = (self.api_key or "").strip()
        return not token or token.lower() in _VERTEX_ADC_PLACEHOLDERS

    def _resolve_api_key(self) -> str:
        if self._vertex_token_provider is not None:
            return self._vertex_token_provider.get_access_token()
        return self.api_key

    def _get_requested_output_dim(self, kwargs: Dict[str, Any]) -> int:
        if "output_dim" in kwargs:
            return kwargs["output_dim"] or 0
        if "output_dim" in self.config:
            return self.config["output_dim"] or 0
        return 0

    def _refresh_clients(self, force: bool = False) -> None:
        if self.provider == LLMProvider.DOUBAO.value and self.llm_type == LLMType.EMBEDDING:
            return

        active_api_key = self._resolve_api_key()
        if not force and self.client is not None and self.async_client is not None:
            if active_api_key == self._active_api_key:
                return

        self._active_api_key = active_api_key
        self.client = OpenAI(api_key=active_api_key, base_url=self.base_url, timeout=self.timeout)
        self.async_client = AsyncOpenAI(
            api_key=active_api_key, base_url=self.base_url, timeout=self.timeout
        )

    def generate(self, prompt: str, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.generate_with_messages(messages, **kwargs)

    def generate_with_messages(self, messages: List[Dict[str, Any]], **kwargs):
        if self.llm_type == LLMType.CHAT:
            return self._openai_chat_completion(messages, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM type for message generation: {self.llm_type}")

    async def generate_with_messages_async(self, messages: List[Dict[str, Any]], **kwargs):
        if self.llm_type == LLMType.CHAT:
            return await self._openai_chat_completion_async(messages, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM type for message generation: {self.llm_type}")

    def generate_with_messages_stream(self, messages: List[Dict[str, Any]], **kwargs):
        """Stream generate response"""
        if self.llm_type == LLMType.CHAT:
            return self._openai_chat_completion_stream(messages, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM type for stream generation: {self.llm_type}")

    async def generate_with_messages_stream_async(self, messages: List[Dict[str, Any]], **kwargs):
        """Async stream generate response"""
        if self.llm_type == LLMType.CHAT:
            return self._openai_chat_completion_stream_async(messages, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM type for stream generation: {self.llm_type}")

    def generate_embedding(self, text: str, **kwargs) -> List[float]:
        if self.llm_type == LLMType.EMBEDDING:
            return self._request_embedding(text, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM type for embedding generation: {self.llm_type}")

    async def generate_embedding_async(self, text: str, **kwargs) -> List[float]:
        if self.llm_type == LLMType.EMBEDDING:
            return await self._request_embedding_async(text, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM type for embedding generation: {self.llm_type}")

    def _openai_chat_completion(self, messages: List[Dict[str, Any]], **kwargs):
        import time

        request_start = time.time()
        try:
            self._refresh_clients()
            # Stage: LLM request preparation

            tools = kwargs.get("tools", None)
            thinking = kwargs.get("thinking", None)

            create_params = {
                "model": self.model,
                "messages": messages,
            }
            if tools:
                create_params["tools"] = tools
                create_params["tool_choice"] = "auto"

            if thinking:
                if self.provider == LLMProvider.DOUBAO.value:
                    create_params["extra_body"] = {"thinking": {"type": thinking}}

            # Stage: LLM API call
            api_start = time.time()
            response = self.client.chat.completions.create(**create_params)

            record_processing_stage(
                "chat_cost", int((time.time() - api_start) * 1000), status="success"
            )

            # Stage: Response parsing
            parse_start = time.time()

            # Record token usage
            if hasattr(response, "usage") and response.usage:
                try:
                    from opencontext.monitoring import record_token_usage

                    record_token_usage(
                        model=self.model,
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )
                except ImportError:
                    pass  # Monitoring module not installed or initialized

            return response
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            # Record failure
            try:
                record_processing_stage(
                    "chat_cost", int((time.time() - request_start) * 1000), status="failure"
                )
            except ImportError:
                pass
            raise

    async def _openai_chat_completion_async(self, messages: List[Dict[str, Any]], **kwargs):
        """Async chat completion"""
        import time

        request_start = time.time()
        try:
            self._refresh_clients()
            tools = kwargs.get("tools", None)
            thinking = kwargs.get("thinking", None)

            create_params = {
                "model": self.model,
                "messages": messages,
            }
            if tools:
                create_params["tools"] = tools
                create_params["tool_choice"] = "auto"

            if thinking:
                if self.provider == LLMProvider.DOUBAO.value:
                    create_params["extra_body"] = {"thinking": {"type": thinking}}
            # Stage: LLM API call
            api_start = time.time()
            response = await self.async_client.chat.completions.create(**create_params)

            record_processing_stage(
                "chat_cost", int((time.time() - api_start) * 1000), status="success"
            )

            # Record token usage
            if hasattr(response, "usage") and response.usage:
                try:
                    from opencontext.monitoring import record_token_usage

                    record_token_usage(
                        model=self.model,
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )
                except ImportError:
                    pass  # Monitoring module not installed or initialized

            return response
        except APIError as e:
            logger.exception(f"OpenAI API async error: {e}")
            # Record failure
            try:
                record_processing_stage(
                    "chat_cost", int((time.time() - request_start) * 1000), status="failure"
                )
            except ImportError:
                pass
            raise

    def _openai_chat_completion_stream(self, messages: List[Dict[str, Any]], **kwargs):
        """Sync stream chat completion"""
        try:
            self._refresh_clients()
            tools = kwargs.get("tools", None)
            thinking = kwargs.get("thinking", None)

            create_params = {
                "model": self.model,
                "messages": messages,
                "stream": True,
            }
            if tools:
                create_params["tools"] = tools
                create_params["tool_choice"] = "auto"

            if thinking:
                if self.provider == LLMProvider.DOUBAO.value:
                    create_params["extra_body"] = {"thinking": {"type": thinking}}

            stream = self.client.chat.completions.create(**create_params)
            return stream
        except APIError as e:
            logger.error(f"OpenAI API stream error: {e}")
            raise

    async def _openai_chat_completion_stream_async(self, messages: List[Dict[str, Any]], **kwargs):
        """Async stream chat completion - async generator"""
        try:
            self._refresh_clients()
            tools = kwargs.get("tools", None)
            thinking = kwargs.get("thinking", None)

            create_params = {
                "model": self.model,
                "messages": messages,
                "stream": True,
            }
            if tools:
                create_params["tools"] = tools
                create_params["tool_choice"] = "auto"

            if thinking:
                if self.provider == LLMProvider.DOUBAO.value:
                    create_params["extra_body"] = {"thinking": {"type": thinking}}

            stream = await self.async_client.chat.completions.create(**create_params)

            # Return stream object directly, it's already an async iterator
            async for chunk in stream:
                yield chunk
        except APIError as e:
            logger.error(f"OpenAI API async stream error: {e}")
            raise

    def _request_embedding(self, text: str, **kwargs) -> List[float]:
        try:
            if self.is_vertex_openai:
                return self._request_vertex_embedding(text, **kwargs)

            self._refresh_clients()
            if self.provider != LLMProvider.DOUBAO.value:
                response = self.client.embeddings.create(model=self.model, input=[text])
                embedding = response.data[0].embedding
            else:
                response = self.client.multimodal_embeddings.create(
                    model=self.model, input=[{"type": "text", "text": text}]
                )
                embedding = response.data.embedding

            # Record token usage
            if hasattr(response, "usage") and response.usage:
                try:
                    from opencontext.monitoring import record_token_usage

                    usage = response.usage
                    if isinstance(usage, dict):
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        total_tokens = usage.get("total_tokens", 0)
                    else:
                        prompt_tokens = usage.prompt_tokens
                        total_tokens = usage.total_tokens

                    record_token_usage(
                        model=self.model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,  # embedding has no completion tokens
                        total_tokens=total_tokens,
                    )
                except ImportError:
                    pass  # Monitoring module not installed or initialized

            output_dim = self._get_requested_output_dim(kwargs)
            if output_dim and len(embedding) > output_dim:
                import math

                embedding = embedding[:output_dim]
                norm = math.sqrt(sum(x**2 for x in embedding))
                if norm > 0:
                    embedding = [x / norm for x in embedding]

            return embedding
        except APIError as e:
            logger.error(f"OpenAI API error during embedding: {e}")
            raise

    async def _request_embedding_async(self, text: str, **kwargs) -> List[float]:
        try:
            if self.is_vertex_openai:
                return await asyncio.to_thread(self._request_vertex_embedding, text, **kwargs)

            self._refresh_clients()
            if self.provider == LLMProvider.DOUBAO.value:
                # Only ark has multimodal_embeddings
                response = self.client.multimodal_embeddings.create(
                    model=self.model, input=[{"type": "text", "text": text}]
                )
                embedding = response.data.embedding
            else:
                response = await self.async_client.embeddings.create(model=self.model, input=[text])
                embedding = response.data[0].embedding

            # Record token usage
            if hasattr(response, "usage") and response.usage:
                try:
                    from opencontext.monitoring import record_token_usage

                    usage = response.usage
                    if isinstance(usage, dict):
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        total_tokens = usage.get("total_tokens", 0)
                    else:
                        prompt_tokens = usage.prompt_tokens
                        total_tokens = usage.total_tokens

                    record_token_usage(
                        model=self.model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,  # embedding has no completion tokens
                        total_tokens=total_tokens,
                    )
                except ImportError:
                    pass  # Monitoring module not installed or initialized

            output_dim = self._get_requested_output_dim(kwargs)
            if output_dim and len(embedding) > output_dim:
                import math

                embedding = embedding[:output_dim]
                norm = math.sqrt(sum(x**2 for x in embedding))
                if norm > 0:
                    embedding = [x / norm for x in embedding]

            return embedding
        except APIError as e:
            logger.error(f"OpenAI API error during embedding: {e}")
            raise

    def _request_vertex_embedding(self, text: str, **kwargs) -> List[float]:
        """Call Vertex's native predict endpoint for embedding models."""
        predict_url = build_vertex_predict_url(self.base_url, self.model)
        access_token = self._resolve_api_key()

        instance = {"content": text}
        task_type = kwargs.get("task_type", self.config.get("task_type"))
        title = kwargs.get("title", self.config.get("title"))
        if task_type:
            instance["task_type"] = task_type
        if title:
            instance["title"] = title

        parameters = {}
        output_dim = self._get_requested_output_dim(kwargs)
        if output_dim:
            parameters["outputDimensionality"] = output_dim

        auto_truncate = kwargs.get("auto_truncate", self.config.get("auto_truncate"))
        if auto_truncate is not None:
            parameters["autoTruncate"] = auto_truncate

        payload = {"instances": [instance]}
        if parameters:
            payload["parameters"] = parameters

        request = urllib_request.Request(
            predict_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib_request.urlopen(request, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Vertex embedding request failed with HTTP {exc.code}: {error_body}"
            ) from exc

        predictions = response_data.get("predictions") or []
        if not predictions:
            raise RuntimeError("Vertex embedding model returned empty predictions")

        embeddings = predictions[0].get("embeddings") or {}
        values = embeddings.get("values")
        if not values:
            raise RuntimeError("Vertex embedding model returned empty embedding values")

        return values

    def vectorize(self, vectorize: Vectorize, **kwargs):
        if vectorize.vector:
            return
        vectorize.vector = self.generate_embedding(vectorize.get_vectorize_content(), **kwargs)
        return

    async def vectorize_async(self, vectorize: Vectorize, **kwargs):
        if vectorize.vector:
            return
        vectorize.vector = await self.generate_embedding_async(
            vectorize.get_vectorize_content(), **kwargs
        )
        return

    def validate(self) -> tuple[bool, str]:
        """
        Validate LLM configuration by making a simple API call.

        Returns:
            tuple[bool, str]: (success, message)
        """

        def _extract_error_summary(error: Any) -> str:
            """
            Extract a concise error summary from API error messages.
            Removes verbose API error details and keeps only the essential information.
            """
            error_msg = str(error)
            if not error_msg:
                return "Unknown error"

            # 1. Check for specific Volcengine/Doubao error codes
            volcengine_errors = {
                "AccessDenied": "Access denied. Please ensure the model is enabled in the Volcengine console.",
                "QuotaExceeded": "Quota exceeded. Please check your Volcengine account balance.",
                "ModelAccountIpmRateLimitExceeded": "Model rate limit (IPM) exceeded.",
                "AccountRateLimitExceeded": "Account rate limit exceeded.",
                "RateLimitExceeded": "Rate limit exceeded.",
                "InternalServiceError": "Volcengine internal service error.",
                "ServiceUnavailable": "Service unavailable.",
                "MethodNotAllowed": "Method not allowed. Check your configuration.",
            }

            for code, msg in volcengine_errors.items():
                if code in error_msg:
                    return msg

            # 2. Check for OpenAI specific errors
            openai_errors = {
                "insufficient_quota": "Insufficient quota. Check your plan and billing details.",
                "invalid_api_key": "Invalid API key provided.",
                "model_not_found": "The model does not exist or you do not have access to it.",
                "context_length_exceeded": "Context length exceeded.",
                "Malformed publisher model": "Vertex AI requires publisher-qualified model names, e.g. google/gemini-2.5-flash.",
            }

            for code, msg in openai_errors.items():
                if code in error_msg:
                    return msg

            # If it's an API error with detailed JSON response, extract key info
            if "Error code:" in error_msg:
                parts = error_msg.split("Error code:", 1)
                if len(parts) > 1:
                    code_part = parts[1].strip()
                    # Get just the code number and basic message
                    if "-" in code_part:
                        code = code_part.split("-", 1)[0].strip()
                        # Try to extract the error type/message from the dict
                        if "'message':" in code_part:
                            try:
                                msg_start = code_part.find("'message':") + len("'message':")
                                msg_part = code_part[msg_start:].strip()
                                if msg_part.startswith("'") or msg_part.startswith('"'):
                                    quote_char = msg_part[0]
                                    msg_end = msg_part.find(quote_char, 1)
                                    if msg_end > 0:
                                        actual_msg = msg_part[1:msg_end]
                                        # Remove Request id and everything after it
                                        if ". Request id:" in actual_msg:
                                            actual_msg = actual_msg.split(". Request id:")[0]
                                        return actual_msg
                            except Exception:
                                pass
                        return f"Error {code}"

            # If the message is already concise (< 150 chars), return as-is
            if len(error_msg) < 150:
                return error_msg

            # Otherwise, truncate with ellipsis
            return error_msg[:147] + "..."

        try:
            self._refresh_clients()
            if self.llm_type == LLMType.CHAT:
                # Test with an image input - 20x20 pixel PNG with clear red square pattern
                # This is a small but visible test image to validate vision capabilities
                # tiny_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAYAAACNiR0NAAAAMElEQVR42mP8z8DwHwMxgImBQjDwBo4aNWrUqFGjRlEEhtEwHDVq1KhRo0aNGgUAAN0/Af9dX6MgAAAAAElFTkSuQmCC"
                # messages = [
                #     {
                #         "role": "user",
                #         "content": [
                #             {"type": "text", "text": "Hi"},
                #             {
                #                 "type": "image_url",
                #                 "image_url": {"url": f"data:image/png;base64,{tiny_image_base64}"},
                #             },
                #         ],
                #     }
                # ]
                messages = [{"role": "user", "content": "Hi"}]
                response = self.client.chat.completions.create(model=self.model, messages=messages)
                if response.choices and len(response.choices) > 0:
                    return True, "Chat model validation successful"
                else:
                    return False, "Chat model returned empty response"

            elif self.llm_type == LLMType.EMBEDDING:
                # Test with a simple text
                if self.provider == LLMProvider.DOUBAO.value:
                    response = self.client.multimodal_embeddings.create(
                        model=self.model, input=[{"type": "text", "text": "test"}]
                    )
                    if response.data and response.data.embedding:
                        return True, "Embedding model validation successful"
                    else:
                        return False, "Embedding model returned empty response"
                elif self.is_vertex_openai:
                    embedding = self._request_vertex_embedding("test")
                    if embedding:
                        return True, "Embedding model validation successful"
                    return False, "Embedding model returned empty response"
                else:
                    response = self.client.embeddings.create(model=self.model, input=["test"])
                    if response.data and len(response.data) > 0 and response.data[0].embedding:
                        return True, "Embedding model validation successful"
                    else:
                        return False, "Embedding model returned empty response"
            else:
                return False, f"Unsupported LLM type: {self.llm_type}"

        except APIError as e:
            logger.error(f"LLM validation failed with API error: {e}")
            # Extract concise error summary before returning
            concise_error = _extract_error_summary(e)
            return False, concise_error
        except Exception as e:
            logger.error(f"LLM validation failed with unexpected error: {e}")
            # Extract concise error summary before returning
            concise_error = _extract_error_summary(e)
            return False, concise_error
