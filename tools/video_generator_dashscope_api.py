from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, List, Optional

import aiohttp

from interfaces.video_output import VideoOutput
from tools.dashscope_common import compact_response, raise_for_provider_status
from utils.image import image_path_to_b64
from utils.rate_limiter import RateLimiter


VIDEO_GENERATION_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"
IMAGE2VIDEO_PATH = "/api/v1/services/aigc/image2video/video-synthesis"


class VideoGeneratorDashScopeAPI:
    def __init__(
        self,
        api_key: str,
        t2v_model: str = "wanx2.1-t2v-turbo",
        ff2v_model: str = "wanx2.1-i2v-turbo",
        flf2v_model: str = "wanx2.1-kf2v-plus",
        base_url: str = "https://dashscope.aliyuncs.com",
        poll_interval: float = 15.0,
        max_poll_attempts: Optional[int] = None,
        max_rate_limit_retries: int = 8,
        rate_limit_retry_base_delay: float = 60.0,
        rate_limit_retry_max_delay: float = 300.0,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.api_key = api_key
        self.t2v_model = t2v_model
        self.ff2v_model = ff2v_model
        self.flf2v_model = flf2v_model
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.max_poll_attempts = max_poll_attempts
        self.max_rate_limit_retries = max_rate_limit_retries
        self.rate_limit_retry_base_delay = rate_limit_retry_base_delay
        self.rate_limit_retry_max_delay = rate_limit_retry_max_delay
        self.rate_limiter = rate_limiter

    async def generate_single_video(
        self,
        prompt: str,
        reference_image_paths: List[str],
        aspect_ratio: str = "16:9",
        duration: int = 5,
        **kwargs,
    ) -> VideoOutput:
        progress = kwargs.get("progress")
        task_id = await self.create_video_generation_task(
            prompt=prompt,
            reference_image_paths=reference_image_paths,
            aspect_ratio=aspect_ratio,
            duration=duration,
            progress=progress,
        )
        video_url = await self.query_video_generation_task(task_id, progress=progress)
        return VideoOutput(fmt="url", ext="mp4", data=video_url)

    async def create_video_generation_task(
        self,
        prompt: str,
        reference_image_paths: List[str],
        aspect_ratio: str = "16:9",
        duration: int = 5,
        progress=None,
    ) -> str:
        payload = self._build_payload(prompt, reference_image_paths, aspect_ratio, duration)
        endpoint_path = _endpoint_path(len(reference_image_paths))
        logging.info("Calling %s to generate video...", payload["model"])
        retry_count = 0
        while True:
            if self.rate_limiter:
                await self.rate_limiter.acquire()
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}{endpoint_path}",
                    headers=self._headers(async_request=True),
                    json=payload,
                ) as response:
                    response_json = await response.json(content_type=None)
                    if response.status == 429 and retry_count < self.max_rate_limit_retries:
                        retry_count += 1
                        delay = self._rate_limit_retry_delay(retry_count, response)
                        _emit_progress(progress, "video_rate_limited", "DashScope video task creation rate-limited; retrying", {"attempt": retry_count, "retry_after_seconds": delay, "model": payload["model"]})
                        await asyncio.sleep(delay)
                        continue
                    raise_for_provider_status("DashScope video task creation", response.status, response_json)
            break
        task_id = _task_id_from_response(response_json)
        if not task_id:
            raise RuntimeError(f"DashScope video task response missing task_id: {compact_response(response_json)}")
        return task_id

    async def query_video_generation_task(self, task_id: str, progress=None) -> str:
        attempts = 0
        rate_limit_retries = 0
        while True:
            if self.max_poll_attempts is not None and attempts >= self.max_poll_attempts:
                raise TimeoutError(f"DashScope video task {task_id} did not complete after {attempts} polls.")
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/v1/tasks/{task_id}",
                    headers=self._headers(async_request=False),
                ) as response:
                    response_json = await response.json(content_type=None)
                    if response.status == 429 and rate_limit_retries < self.max_rate_limit_retries:
                        rate_limit_retries += 1
                        delay = self._rate_limit_retry_delay(rate_limit_retries, response)
                        _emit_progress(progress, "video_rate_limited", "DashScope video task query rate-limited; retrying", {"attempt": rate_limit_retries, "retry_after_seconds": delay, "task_id": task_id})
                        await asyncio.sleep(delay)
                        continue
                    raise_for_provider_status("DashScope video task query", response.status, response_json)
            attempts += 1
            rate_limit_retries = 0
            output = response_json.get("output") if isinstance(response_json, dict) else None
            if not isinstance(output, dict):
                raise RuntimeError(f"DashScope video task query missing output: {compact_response(response_json)}")
            status = str(output.get("task_status") or output.get("status") or "").upper()
            if status in {"SUCCEEDED", "SUCCESS", "COMPLETED"}:
                video_url = output.get("video_url") or output.get("url")
                if not video_url:
                    raise RuntimeError(f"DashScope video task succeeded without video_url: {compact_response(response_json)}")
                return str(video_url)
            if status in {"FAILED", "CANCELED", "CANCELLED"}:
                message = output.get("message") or output.get("code") or compact_response(response_json)
                raise RuntimeError(f"DashScope video task failed: {message}")
            await asyncio.sleep(self.poll_interval)

    def _rate_limit_retry_delay(self, attempt: int, response: Any) -> float:
        retry_after = getattr(response, "headers", {}).get("Retry-After") if getattr(response, "headers", None) else None
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        return min(self.rate_limit_retry_base_delay * attempt, self.rate_limit_retry_max_delay)

    def _build_payload(self, prompt: str, reference_image_paths: List[str], aspect_ratio: str, duration: int) -> dict[str, Any]:
        if len(reference_image_paths) == 0:
            model = self.t2v_model
            input_payload: dict[str, Any] = {"prompt": prompt}
        elif len(reference_image_paths) == 1:
            model = self.ff2v_model
            input_payload = {"prompt": prompt, "img_url": _image_uri(reference_image_paths[0])}
        elif len(reference_image_paths) == 2:
            model = self.flf2v_model
            input_payload = {
                "prompt": prompt,
                "first_frame_url": _image_uri(reference_image_paths[0]),
                "last_frame_url": _image_uri(reference_image_paths[1]),
            }
        else:
            raise ValueError("DashScope video generation supports at most first and last frame images")

        parameters: dict[str, Any] = {"prompt_extend": True}
        if len(reference_image_paths) == 0:
            parameters["size"] = _size_from_aspect_ratio(aspect_ratio)
            parameters["duration"] = duration
        elif len(reference_image_paths) == 1:
            parameters["resolution"] = "720P"
            parameters["duration"] = duration
        else:
            parameters["resolution"] = "720P"

        return {
            "model": model,
            "input": input_payload,
            "parameters": parameters,
        }

    def _headers(self, async_request: bool) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if async_request:
            headers["X-DashScope-Async"] = "enable"
        return headers


def _task_id_from_response(response_json: dict[str, Any]) -> str:
    output = response_json.get("output") if isinstance(response_json, dict) else None
    if isinstance(output, dict) and output.get("task_id"):
        return str(output["task_id"])
    if isinstance(response_json, dict) and response_json.get("task_id"):
        return str(response_json["task_id"])
    return ""


def _endpoint_path(frame_count: int) -> str:
    if frame_count == 2:
        return IMAGE2VIDEO_PATH
    return VIDEO_GENERATION_PATH


def _image_uri(image_path: str) -> str:
    if image_path.startswith(("http://", "https://", "data:")):
        return image_path

    source_url_path = f"{image_path}.source_url"
    if os.path.exists(source_url_path):
        with open(source_url_path, "r", encoding="utf-8") as handle:
            source_url = handle.read().strip()
        if source_url.startswith(("http://", "https://", "data:")):
            return source_url

    return image_path_to_b64(image_path, mime=True)


def _size_from_aspect_ratio(aspect_ratio: str) -> str:
    if aspect_ratio == "9:16":
        return "720*1280"
    if aspect_ratio == "1:1":
        return "1024*1024"
    return "1280*720"


def _emit_progress(progress, stage: str, message: str, metadata: dict[str, Any]) -> None:
    if progress is not None:
        progress(stage, message, metadata)
