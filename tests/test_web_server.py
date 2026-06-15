from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient` is deprecated.*")

from fastapi.testclient import TestClient

from web_server import (
    VideoJobManager,
    create_app,
    diagnose_error,
    format_sse,
    validate_job_payload,
    _preferred_config,
)


class PayloadValidationTests(unittest.TestCase):
    def test_idea_job_requires_idea_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "idea is required"):
            validate_job_payload({"mode": "idea2video", "idea": "", "user_requirement": "short", "style": "Anime"})

    def test_script_job_requires_script_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "script is required"):
            validate_job_payload({"mode": "script2video", "script": "", "user_requirement": "short", "style": "Anime"})

    def test_validation_normalizes_mode_and_text_fields(self) -> None:
        payload = validate_job_payload(
            {
                "mode": "idea2video",
                "idea": "  a cat opens a bakery  ",
                "user_requirement": "  3 scenes max ",
                "style": " watercolor ",
            }
        )

        self.assertEqual(payload["mode"], "idea2video")
        self.assertEqual(payload["idea"], "a cat opens a bakery")
        self.assertEqual(payload["user_requirement"], "3 scenes max")
        self.assertEqual(payload["style"], "watercolor")


class SseFormattingTests(unittest.TestCase):
    def test_format_sse_uses_event_type_and_json_data(self) -> None:
        payload = {"type": "progress", "message": "Rendering", "metadata": {"stage": "frames"}}

        event = format_sse(payload)

        self.assertTrue(event.startswith("event: progress\n"))
        self.assertIn('"message": "Rendering"', event)
        self.assertTrue(event.endswith("\n\n"))


class ConfigSelectionTests(unittest.TestCase):
    def test_preferred_config_uses_local_file_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "idea2video.yaml"
            local = Path(tmp) / "idea2video.local.yaml"
            config.write_text("base: true\n", encoding="utf-8")
            local.write_text("local: true\n", encoding="utf-8")

            self.assertEqual(_preferred_config(str(config)), local)

    def test_preferred_config_falls_back_to_base_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "script2video.yaml"
            config.write_text("base: true\n", encoding="utf-8")

            self.assertEqual(_preferred_config(str(config)), config)


class ErrorDiagnosisTests(unittest.TestCase):
    def test_image_reference_limit_error_lists_specific_cause(self) -> None:
        details = diagnose_error(ValueError("DashScope image edit supports at most 3 reference images"))

        self.assertTrue(any("3 张" in cause for cause in details["possible_causes"]))
        self.assertTrue(any("参考图" in step for step in details["next_steps"]))

    def test_dashscope_url_error_lists_possible_causes(self) -> None:
        details = diagnose_error(
            RuntimeError(
                "DashScope video task creation failed with HTTP 400: url error, please check url; "
                'response={"code":"InvalidParameter"}'
            )
        )

        self.assertTrue(details["possible_causes"])
        self.assertTrue(any("图片 URL" in cause for cause in details["possible_causes"]))
        self.assertTrue(details["evidence"])
        self.assertTrue(details["next_steps"])

    def test_dashscope_rate_quota_error_lists_rate_limit_cause(self) -> None:
        details = diagnose_error(
            RuntimeError(
                "DashScope video task creation failed with HTTP 429: Requests rate limit exceeded; "
                'response={"code":"Throttling.RateQuota"}'
            )
        )

        self.assertTrue(any("请求频率" in cause for cause in details["possible_causes"]))
        self.assertFalse(any("无效令牌" in cause for cause in details["possible_causes"]))
        self.assertTrue(any("429" in item for item in details["evidence"]))


class VideoJobManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_job_runs_pipeline_and_records_completed_status(self) -> None:
        async def fake_runner(job, emit):
            emit("progress", "Planning shots", {"stage": "storyboard"})
            output = Path(job.working_dir) / "final_video.mp4"
            output.write_bytes(b"video")
            return output

        with tempfile.TemporaryDirectory() as tmp:
            manager = VideoJobManager(Path(tmp), runner=fake_runner)
            job = await manager.create_job(
                {
                    "mode": "script2video",
                    "script": "INT. ROOM - DAY\nA test scene.",
                    "user_requirement": "No more than 1 shot.",
                    "style": "Realistic",
                }
            )

            await manager.wait_for_job(job.job_id)
            snapshot = manager.snapshot(job.job_id)

        self.assertEqual(snapshot["status"], "completed")
        self.assertEqual(snapshot["mode"], "script2video")
        self.assertTrue(snapshot["video_ready"])
        self.assertEqual(snapshot["events"][-1]["type"], "completed")

    async def test_create_job_records_failed_status_on_runner_error(self) -> None:
        async def failing_runner(job, emit):
            raise RuntimeError("DashScope video task creation failed with HTTP 400: url error, please check url")

        with tempfile.TemporaryDirectory() as tmp:
            manager = VideoJobManager(Path(tmp), runner=failing_runner)
            job = await manager.create_job(
                {
                    "mode": "idea2video",
                    "idea": "A tiny robot learns to paint.",
                    "user_requirement": "One scene.",
                    "style": "Studio lighting",
                }
            )

            await manager.wait_for_job(job.job_id)
            snapshot = manager.snapshot(job.job_id)

        self.assertEqual(snapshot["status"], "failed")
        self.assertIn("url error", snapshot["error"])
        self.assertIn("possible_causes", snapshot["error_details"])
        self.assertTrue(snapshot["error_details"]["possible_causes"])
        self.assertEqual(snapshot["events"][-1]["type"], "failed")
        self.assertEqual(snapshot["events"][-1]["metadata"], snapshot["error_details"])


class WebApiTests(unittest.TestCase):
    def setUp(self) -> None:
        async def fake_runner(job, emit):
            output = Path(job.working_dir) / "final_video.mp4"
            output.write_bytes(b"video")
            return output

        self.tmp = tempfile.TemporaryDirectory()
        self.manager = VideoJobManager(Path(self.tmp.name), runner=fake_runner)
        self.client = TestClient(create_app(manager=self.manager, static_dir=Path("web")))

    def tearDown(self) -> None:
        self.client.close()
        self.tmp.cleanup()

    def test_post_jobs_creates_background_job(self) -> None:
        response = self.client.post(
            "/api/jobs",
            json={
                "mode": "idea2video",
                "idea": "A lighthouse wakes up.",
                "user_requirement": "Two scenes.",
                "style": "Cinematic",
            },
        )

        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertIn("job_id", payload)
        self.assertEqual(payload["status"], "queued")

    def test_post_jobs_returns_validation_error(self) -> None:
        response = self.client.post("/api/jobs", json={"mode": "idea2video"})

        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload["error"], "idea is required for idea2video jobs")

    def test_index_serves_web_console(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn("text2video 视频控制台", body)
        self.assertIn("小狗在草地上追蝴蝶", body)

    def test_video_endpoint_serves_completed_job_video(self) -> None:
        async def create_completed_job() -> str:
            job = await self.manager.create_job(
                {
                    "mode": "script2video",
                    "script": "INT. ROOM - DAY\nA test scene.",
                    "user_requirement": "One shot.",
                    "style": "Realistic",
                }
            )
            await self.manager.wait_for_job(job.job_id)
            return job.job_id

        job_id = asyncio.run(create_completed_job())

        response = self.client.get(f"/api/jobs/{job_id}/video")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "video/mp4")
        self.assertEqual(response.content, b"video")
