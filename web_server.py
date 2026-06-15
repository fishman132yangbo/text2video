from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from langchain.chat_models import init_chat_model

from pipelines.idea2video_pipeline import Idea2VideoPipeline
from pipelines.script2video_pipeline import Script2VideoPipeline
from tools.render_backend import RenderBackend
from utils.provider_presets import resolve_chat_model_config


JobRunner = Callable[["VideoJob", Callable[[str, str, dict[str, Any] | None], None]], Awaitable[Path]]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class VideoJob:
    job_id: str
    mode: str
    payload: dict[str, str]
    working_dir: Path
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    error: str = ""
    error_details: dict[str, Any] = field(default_factory=dict)
    final_video_path: Path | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[asyncio.Queue[dict[str, Any]]] = field(default_factory=list)
    task: asyncio.Task[None] | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "mode": self.mode,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "error_details": self.error_details,
            "working_dir": str(self.working_dir),
            "video_ready": bool(self.final_video_path and self.final_video_path.exists()),
            "events": self.events,
        }


class VideoJobManager:
    def __init__(self, jobs_root: Path, runner: JobRunner | None = None) -> None:
        self.jobs_root = Path(jobs_root)
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.runner = runner or run_video_pipeline
        self.jobs: dict[str, VideoJob] = {}

    async def create_job(self, payload: dict[str, Any]) -> VideoJob:
        normalized = validate_job_payload(payload)
        job_id = uuid.uuid4().hex[:12]
        working_dir = self.jobs_root / job_id
        working_dir.mkdir(parents=True, exist_ok=True)
        job = VideoJob(job_id=job_id, mode=normalized["mode"], payload=normalized, working_dir=working_dir)
        self.jobs[job_id] = job
        self.emit(job, "queued", "Job queued", {"mode": job.mode})
        job.task = asyncio.create_task(self._run_job(job))
        return job

    async def wait_for_job(self, job_id: str) -> None:
        job = self.require_job(job_id)
        if job.task is not None:
            await job.task

    def require_job(self, job_id: str) -> VideoJob:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def snapshot(self, job_id: str) -> dict[str, Any]:
        return self.require_job(job_id).snapshot()

    def emit(self, job: VideoJob, event_type: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        event = {
            "type": event_type,
            "message": message,
            "metadata": metadata or {},
            "created_at": _now(),
        }
        job.events.append(event)
        job.updated_at = event["created_at"]
        for queue in list(job.subscribers):
            queue.put_nowait(event)

    async def _run_job(self, job: VideoJob) -> None:
        job.status = "running"
        self.emit(job, "running", "Job started", {"mode": job.mode})

        def progress(stage: str, message: str, metadata: dict[str, Any] | None = None) -> None:
            payload = dict(metadata or {})
            payload["stage"] = stage
            self.emit(job, "progress", message, payload)

        try:
            final_video = await self.runner(job, progress)
            job.final_video_path = Path(final_video)
            job.status = "completed"
            self.emit(job, "completed", "Video generation completed", {"video_path": str(final_video)})
        except Exception as exc:  # pragma: no cover - exact exceptions depend on providers.
            job.status = "failed"
            job.error = str(exc)
            job.error_details = diagnose_error(exc)
            self.emit(job, "failed", str(exc), job.error_details)

    def subscribe(self, job: VideoJob) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        job.subscribers.append(queue)
        return queue

    def unsubscribe(self, job: VideoJob, queue: asyncio.Queue[dict[str, Any]]) -> None:
        if queue in job.subscribers:
            job.subscribers.remove(queue)


def validate_job_payload(payload: dict[str, Any]) -> dict[str, str]:
    mode = str(payload.get("mode") or "").strip()
    if mode not in {"idea2video", "script2video"}:
        raise ValueError("mode must be idea2video or script2video")

    user_requirement = str(payload.get("user_requirement") or "").strip()
    style = str(payload.get("style") or "").strip() or "Realistic"
    normalized = {"mode": mode, "user_requirement": user_requirement, "style": style}

    if mode == "idea2video":
        idea = str(payload.get("idea") or "").strip()
        if not idea:
            raise ValueError("idea is required for idea2video jobs")
        normalized["idea"] = idea
    else:
        script = str(payload.get("script") or "").strip()
        if not script:
            raise ValueError("script is required for script2video jobs")
        normalized["script"] = script
    return normalized


def format_sse(event: dict[str, Any]) -> str:
    event_type = str(event.get("type") or "message")
    data = json.dumps(event, ensure_ascii=False)
    return f"event: {event_type}\ndata: {data}\n\n"


def diagnose_error(exc: BaseException) -> dict[str, Any]:
    message = str(exc)
    lower = message.lower()
    possible_causes: list[str] = []
    evidence: list[str] = []
    next_steps: list[str] = []

    if "image edit supports at most 3 reference images" in lower:
        evidence.append("DashScope Qwen Image Edit 只允许最多 3 张参考图，本次本地参数超过了这个上限。")
        possible_causes.extend(
            [
                "参考图选择器选中了超过 3 张图片，包括角色多视角图、已有镜头图或新相机参考图。",
                "旧任务目录里已有 selector_output.json，里面保存了超限的参考图选择结果。",
                "当前图片模型的参考图上限低于其他图片生成模型。",
            ]
        )
        next_steps.extend(
            [
                "重新提交新任务，使用已限制参考图数量的后端代码。",
                "如果复用旧 working_dir，删除或重新生成超限的 *_selector_output.json。",
                "检查任务 events 中的 frame_prompt_limited 事件，确认参考图已被压到模型上限。",
            ]
        )
    elif "dashscope" in lower and "url error" in lower:
        evidence.append("DashScope 返回 HTTP 400 InvalidParameter，并明确提示 url error。")
        possible_causes.extend(
            [
                "传给视频接口的图片 URL 不可访问、已过期，或不是公网 HTTPS 地址。",
                "首尾帧视频使用了不匹配的端点、模型或参数。",
                "本地图片被转成 data URL 后仍不被当前视频模型接受。",
            ]
        )
        next_steps.extend(
            [
                "确认新任务使用的是最新后端代码，并重新提交任务。",
                "检查生成帧旁边是否存在 .source_url 文件，优先使用图片生成接口返回的原始 URL。",
                "如果仍失败，保留 job_id 和 DashScope request_id 用于定位具体请求。",
            ]
        )
    elif "dashscope" in lower and "http 404" in lower:
        evidence.append("DashScope 返回 HTTP 404。")
        possible_causes.extend(
            [
                "base_url 或原生 API 路径配置错误。",
                "模型名在当前账号、区域或接口下不可用。",
                "把兼容 OpenAI 的地址用于 DashScope 原生图像/视频接口。",
            ]
        )
        next_steps.extend(["核对 base_url 是否为 https://dashscope.aliyuncs.com。", "核对模型名是否为当前账号可调用的 DashScope 模型。"])
    elif "throttling.ratequota" in lower or "requests rate limit exceeded" in lower:
        evidence.append("DashScope 返回 HTTP 429 Throttling.RateQuota，表示请求频率超过当前账号或模型限额。")
        possible_causes.extend(
            [
                "视频任务创建或查询请求频率过高。",
                "多个镜头视频同时生成，导致 DashScope 同时处理和轮询请求过多。",
                "当前账号、模型或区域的视频生成限额低于本项目默认请求节奏。",
            ]
        )
        next_steps.extend(
            [
                "使用已带 429 自动等待重试的后端代码重新提交任务。",
                "减少同时运行的 Web 任务数量，不要并行提交多个视频生成任务。",
                "如果仍频繁触发，继续降低 video_generator.max_requests_per_minute 或升级 DashScope 配额。",
            ]
        )
    elif "http 429" in lower or "rate limit" in lower or "无效令牌" in message:
        evidence.append("服务商返回限流、冷却或令牌异常信息。")
        possible_causes.extend(
            [
                "API Key 无效或多次使用无效令牌触发冷却。",
                "请求频率或日额度超过服务商限制。",
                "账号没有开通对应模型或余额不足。",
            ]
        )
        next_steps.extend(["等待服务商提示的冷却时间后重试。", "核对本地配置里的 API Key 和模型权限。", "降低并发或调低配置里的 rate limit。"])
    elif "retryerror" in lower and "keyerror" in lower:
        evidence.append("任务在重试后仍因 KeyError 失败。")
        possible_causes.extend(
            [
                "服务商响应结构与当前解析逻辑不一致。",
                "接口返回错误体但上层代码按成功体读取字段。",
                "模型或端点不匹配导致缺少预期字段。",
            ]
        )
        next_steps.extend(["查看任务 events 中最后一个 provider 响应。", "检查对应适配器的响应字段解析。"])

    if not possible_causes:
        possible_causes.append("上游模型服务、网络请求、配置或本地文件状态异常。")
    if not evidence:
        evidence.append(message or exc.__class__.__name__)
    if not next_steps:
        next_steps.extend(["查看任务 events 和 working_dir 下的中间文件。", "使用完整错误信息和 job_id 继续定位。"])

    return {
        "message": message,
        "possible_causes": possible_causes,
        "evidence": evidence,
        "next_steps": next_steps,
    }


async def run_video_pipeline(job: VideoJob, emit: Callable[[str, str, dict[str, Any] | None], None]) -> Path:
    if job.mode == "idea2video":
        emit("load_config", "Loading Idea2Video configuration", {})
        pipeline = _build_idea_pipeline(job.working_dir / "idea2video")
        emit("pipeline_start", "Starting Idea2Video pipeline", {})
        final_video = await pipeline(
            idea=job.payload["idea"],
            user_requirement=job.payload["user_requirement"],
            style=job.payload["style"],
            quiet=True,
        )
        return Path(final_video)

    emit("load_config", "Loading Script2Video configuration", {})
    pipeline = _build_script_pipeline(job.working_dir / "script2video")
    emit("pipeline_start", "Starting Script2Video pipeline", {})
    final_video = await pipeline(
        script=job.payload["script"],
        user_requirement=job.payload["user_requirement"],
        style=job.payload["style"],
        quiet=True,
        progress=emit,
    )
    return Path(final_video)


def _build_idea_pipeline(working_dir: Path) -> Idea2VideoPipeline:
    config = _load_config(_preferred_config("configs/idea2video.yaml"), working_dir)
    chat_model, backend = _build_components(config)
    return Idea2VideoPipeline(
        chat_model=chat_model,
        image_generator=backend.image_generator,
        video_generator=backend.video_generator,
        working_dir=config["working_dir"],
    )


def _build_script_pipeline(working_dir: Path) -> Script2VideoPipeline:
    config = _load_config(_preferred_config("configs/script2video.yaml"), working_dir)
    chat_model, backend = _build_components(config)
    return Script2VideoPipeline(
        chat_model=chat_model,
        image_generator=backend.image_generator,
        video_generator=backend.video_generator,
        working_dir=config["working_dir"],
    )


def _preferred_config(config_path: str) -> Path:
    path = Path(config_path)
    local_path = path.with_name(f"{path.stem}.local{path.suffix}")
    return local_path if local_path.exists() else path


def _load_config(config_path: str | Path, working_dir: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["working_dir"] = str(working_dir)
    return config


def _build_components(config: dict[str, Any]) -> tuple[Any, RenderBackend]:
    chat_model_args = resolve_chat_model_config(config["chat_model"]["init_args"])
    chat_model = init_chat_model(**chat_model_args)
    backend = RenderBackend.from_config(config)
    return chat_model, backend


def create_app(manager: VideoJobManager | None = None, static_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="text2video Web Video Console")
    app.state.manager = manager or VideoJobManager(Path(".working_dir/web_jobs"))
    app.state.static_dir = Path(static_dir or "web")

    app.add_api_route("/api/jobs", create_job_handler, methods=["POST"], response_model=None)
    app.add_api_route("/api/jobs/{job_id}", get_job_handler, methods=["GET"], response_model=None)
    app.add_api_route("/api/jobs/{job_id}/events", job_events_handler, methods=["GET"], response_model=None)
    app.add_api_route("/api/jobs/{job_id}/video", job_video_handler, methods=["GET"], response_model=None)
    app.add_api_route("/", index_handler, methods=["GET"], response_model=None)
    app.add_api_route("/{name:path}", static_handler, methods=["GET"], response_model=None)
    return app


def _manager(request: Request) -> VideoJobManager:
    return request.app.state.manager


def _static_dir(request: Request) -> Path:
    return request.app.state.static_dir


async def create_job_handler(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "request body must be JSON"}, status_code=400)
    try:
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        job = await _manager(request).create_job(payload)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse(job.snapshot(), status_code=202)


async def get_job_handler(request: Request) -> JSONResponse:
    try:
        return JSONResponse(_manager(request).snapshot(request.path_params["job_id"]))
    except KeyError:
        return JSONResponse({"error": "job not found"}, status_code=404)


async def job_events_handler(request: Request) -> StreamingResponse | PlainTextResponse:
    manager = _manager(request)
    try:
        job = manager.require_job(request.path_params["job_id"])
    except KeyError:
        return PlainTextResponse("job not found", status_code=404)

    async def event_stream():
        for event in job.events:
            yield format_sse(event)
        if job.status in {"completed", "failed"}:
            return

        queue = manager.subscribe(job)
        try:
            while True:
                event = await queue.get()
                yield format_sse(event)
                if event.get("type") in {"completed", "failed"}:
                    break
        finally:
            manager.unsubscribe(job, queue)

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


async def job_video_handler(request: Request) -> FileResponse | PlainTextResponse:
    try:
        job = _manager(request).require_job(request.path_params["job_id"])
    except KeyError:
        return PlainTextResponse("job not found", status_code=404)
    if not job.final_video_path or not job.final_video_path.exists():
        return PlainTextResponse("video not ready", status_code=404)
    return FileResponse(job.final_video_path, media_type="video/mp4")


async def index_handler(request: Request) -> FileResponse | PlainTextResponse:
    index_path = _static_dir(request) / "index.html"
    if not index_path.exists():
        return PlainTextResponse("web frontend not found", status_code=404)
    return FileResponse(index_path)


async def static_handler(request: Request) -> FileResponse | PlainTextResponse:
    name = request.path_params["name"].strip("/")
    static_dir = _static_dir(request).resolve()
    candidate = (static_dir / name).resolve()
    if candidate == static_dir or static_dir not in candidate.parents or not candidate.is_file():
        return PlainTextResponse("asset not found", status_code=404)
    return FileResponse(candidate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the text2video web video console.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7860, type=int)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
