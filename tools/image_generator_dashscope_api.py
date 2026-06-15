from __future__ import annotations

import logging
from typing import Any, List, Optional

import aiohttp
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt

from interfaces.image_output import ImageOutput
from tools.dashscope_common import NonRetryableProviderError, compact_response, raise_for_provider_status
from utils.image import image_path_to_b64
from utils.rate_limiter import RateLimiter
from utils.retry import after_func


class ImageGeneratorDashScopeAPI:
    max_reference_images = 3

    def __init__(
        self,
        api_key: str,
        model: str = "qwen-image",
        edit_model: str = "qwen-image-edit",
        base_url: str = "https://dashscope.aliyuncs.com",
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.edit_model = edit_model
        self.base_url = _normalize_base_url(base_url)
        self.rate_limiter = rate_limiter

    @retry(stop=stop_after_attempt(3), after=after_func, reraise=True, retry=retry_if_not_exception_type(NonRetryableProviderError))
    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: List[str] = [],
        aspect_ratio: Optional[str] = "16:9",
        size: Optional[str] = None,
        **kwargs,
    ) -> ImageOutput:
        logging.info("Calling %s to generate image...", self.model if not reference_image_paths else self.edit_model)
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        if reference_image_paths:
            return await self._edit_image(prompt, reference_image_paths, size=size)
        return await self._generate_image(prompt, size=size or _size_from_aspect_ratio(aspect_ratio))

    async def _generate_image(self, prompt: str, size: str) -> ImageOutput:
        payload = _build_generation_payload(self.model, prompt, [], size)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/services/aigc/multimodal-generation/generation",
                headers=self._headers(),
                json=payload,
            ) as response:
                response_json = await response.json(content_type=None)
                raise_for_provider_status("DashScope image generation", response.status, response_json)
        return _extract_image_output(response_json)

    async def _edit_image(self, prompt: str, reference_image_paths: List[str], size: Optional[str]) -> ImageOutput:
        if len(reference_image_paths) > 3:
            raise ValueError("DashScope image edit supports at most 3 reference images")
        payload = _build_generation_payload(self.edit_model, prompt, reference_image_paths, size)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/services/aigc/multimodal-generation/generation",
                headers=self._headers(),
                json=payload,
            ) as response:
                response_json = await response.json(content_type=None)
                raise_for_provider_status("DashScope image edit", response.status, response_json)
        return _extract_image_output(response_json)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


def _extract_image_output(response_json: dict[str, Any]) -> ImageOutput:
    output = response_json.get("output") if isinstance(response_json, dict) else None
    if isinstance(output, dict):
        choices = output.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("image"), str) and item["image"]:
                        return ImageOutput(fmt="url", ext="png", data=item["image"])

        results = output.get("results")
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and isinstance(first.get("url"), str) and first["url"]:
                return ImageOutput(fmt="url", ext="png", data=first["url"])

    data = response_json.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            if isinstance(first.get("url"), str) and first["url"]:
                return ImageOutput(fmt="url", ext="png", data=first["url"])
            if isinstance(first.get("b64_json"), str) and first["b64_json"]:
                return ImageOutput(fmt="b64", ext="png", data=first["b64_json"])
    raise NonRetryableProviderError(f"DashScope image response missing image URL: {compact_response(response_json)}")


def _size_from_aspect_ratio(aspect_ratio: Optional[str]) -> str:
    if aspect_ratio == "9:16":
        return "928*1664"
    if aspect_ratio == "1:1":
        return "1328*1328"
    return "1664*928"


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    for suffix in ("/compatible-mode/v1", "/api/v1"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _normalize_size(size: Optional[str]) -> Optional[str]:
    if not size:
        return None
    return size.replace("x", "*")


def _build_generation_payload(model: str, prompt: str, reference_image_paths: List[str], size: Optional[str]) -> dict[str, Any]:
    content = [{"image": image_path_to_b64(path, mime=True)} for path in reference_image_paths]
    content.append({"text": prompt})

    parameters: dict[str, Any] = {
        "n": 1,
        "watermark": False,
    }
    if model != "qwen-image-edit":
        parameters["prompt_extend"] = True
        normalized_size = _normalize_size(size)
        if normalized_size:
            parameters["size"] = normalized_size

    return {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        },
        "parameters": parameters,
    }
