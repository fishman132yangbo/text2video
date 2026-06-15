import unittest
import tempfile
from unittest.mock import patch

from interfaces.image_output import ImageOutput
from tools.image_generator_dashscope_api import ImageGeneratorDashScopeAPI
from tools.video_generator_dashscope_api import VideoGeneratorDashScopeAPI


class _FakeResponse:
    def __init__(self, payload, status=200, headers=None):
        self.payload = payload
        self.status = status
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self, content_type=None):
        return self.payload


class _FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        response = self.responses.pop(0)
        payload, status, headers = _response_parts(response)
        return _FakeResponse(payload, status, headers)

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        response = self.responses.pop(0)
        payload, status, headers = _response_parts(response)
        return _FakeResponse(payload, status, headers)


def _response_parts(response):
    if len(response) == 3:
        return response
    payload, status = response
    return payload, status, {}


class _FakeRateLimiter:
    def __init__(self):
        self.calls = 0

    async def acquire(self):
        self.calls += 1


class DashScopeImageGeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_to_image_uses_dashscope_multimodal_generation_endpoint(self):
        session = _FakeSession([({"output": {"choices": [{"message": {"content": [{"image": "https://example.com/image.png"}]}}]}}, 200)])
        generator = ImageGeneratorDashScopeAPI(api_key="test-key", model="qwen-image")

        with patch("tools.image_generator_dashscope_api.aiohttp.ClientSession", return_value=session):
            output = await generator.generate_single_image("画一只猫")

        self.assertEqual(output.fmt, "url")
        self.assertEqual(output.data, "https://example.com/image.png")
        method, url, kwargs = session.calls[0]
        self.assertEqual(method, "post")
        self.assertEqual(url, "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["json"]["model"], "qwen-image")
        self.assertEqual(kwargs["json"]["input"]["messages"][0]["content"][0]["text"], "画一只猫")
        self.assertEqual(kwargs["json"]["parameters"]["size"], "1664*928")

    async def test_image_edit_sends_reference_image_content(self):
        session = _FakeSession([({"output": {"choices": [{"message": {"content": [{"image": "https://example.com/edit.png"}]}}]}}, 200)])
        generator = ImageGeneratorDashScopeAPI(api_key="test-key", edit_model="qwen-image-edit")

        with tempfile.NamedTemporaryFile(suffix=".png") as reference:
            reference.write(b"image-bytes")
            reference.flush()
            with patch("tools.image_generator_dashscope_api.aiohttp.ClientSession", return_value=session):
                output = await generator.generate_single_image("改成侧面", reference_image_paths=[reference.name])

        self.assertEqual(output.data, "https://example.com/edit.png")
        content = session.calls[0][2]["json"]["input"]["messages"][0]["content"]
        self.assertTrue(content[0]["image"].startswith("data:image/png;base64,"))
        self.assertEqual(content[1]["text"], "改成侧面")

    async def test_text_to_image_acquires_rate_limiter(self):
        session = _FakeSession([({"data": [{"url": "https://example.com/image.png"}]}, 200)])
        rate_limiter = _FakeRateLimiter()
        generator = ImageGeneratorDashScopeAPI(api_key="test-key", rate_limiter=rate_limiter)

        with patch("tools.image_generator_dashscope_api.aiohttp.ClientSession", return_value=session):
            await generator.generate_single_image("画一只猫")

        self.assertEqual(rate_limiter.calls, 1)

    async def test_provider_rejection_is_not_retried(self):
        session = _FakeSession([({"error": {"message": "invalid api key"}}, 401)])
        generator = ImageGeneratorDashScopeAPI(api_key="bad-key")

        with patch("tools.image_generator_dashscope_api.aiohttp.ClientSession", return_value=session):
            with self.assertRaisesRegex(RuntimeError, "invalid api key"):
                await generator.generate_single_image("画一只猫")

        self.assertEqual(len(session.calls), 1)


class DashScopeVideoGeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_to_video_creates_task_and_returns_video_url(self):
        session = _FakeSession(
            [
                ({"output": {"task_id": "task-1", "task_status": "PENDING"}}, 200),
                ({"output": {"task_status": "SUCCEEDED", "video_url": "https://example.com/video.mp4"}}, 200),
            ]
        )
        generator = VideoGeneratorDashScopeAPI(api_key="test-key", t2v_model="wanx2.1-t2v-turbo", poll_interval=0, max_poll_attempts=1)

        with patch("tools.video_generator_dashscope_api.aiohttp.ClientSession", return_value=session):
            output = await generator.generate_single_video("镜头缓慢推进", [])

        self.assertEqual(output.fmt, "url")
        self.assertEqual(output.data, "https://example.com/video.mp4")
        create_call = session.calls[0]
        self.assertEqual(create_call[1], "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis")
        self.assertEqual(create_call[2]["json"]["model"], "wanx2.1-t2v-turbo")
        self.assertEqual(create_call[2]["headers"]["X-DashScope-Async"], "enable")
        self.assertEqual(session.calls[1][1], "https://dashscope.aliyuncs.com/api/v1/tasks/task-1")

    async def test_text_to_video_acquires_rate_limiter(self):
        session = _FakeSession(
            [
                ({"output": {"task_id": "task-1", "task_status": "PENDING"}}, 200),
                ({"output": {"task_status": "SUCCEEDED", "video_url": "https://example.com/video.mp4"}}, 200),
            ]
        )
        rate_limiter = _FakeRateLimiter()
        generator = VideoGeneratorDashScopeAPI(api_key="test-key", poll_interval=0, max_poll_attempts=1, rate_limiter=rate_limiter)

        with patch("tools.video_generator_dashscope_api.aiohttp.ClientSession", return_value=session):
            await generator.generate_single_video("镜头缓慢推进", [])

        self.assertEqual(rate_limiter.calls, 1)

    async def test_video_task_creation_retries_rate_limit_response(self):
        session = _FakeSession(
            [
                ({"code": "Throttling.RateQuota", "message": "Requests rate limit exceeded"}, 429),
                ({"output": {"task_id": "task-1", "task_status": "PENDING"}}, 200),
                ({"output": {"task_status": "SUCCEEDED", "video_url": "https://example.com/video.mp4"}}, 200),
            ]
        )
        generator = VideoGeneratorDashScopeAPI(
            api_key="test-key",
            poll_interval=0,
            max_poll_attempts=1,
            rate_limit_retry_base_delay=0,
            rate_limit_retry_max_delay=0,
        )

        with patch("tools.video_generator_dashscope_api.aiohttp.ClientSession", return_value=session):
            output = await generator.generate_single_video("镜头缓慢推进", [])

        self.assertEqual(output.data, "https://example.com/video.mp4")
        self.assertEqual([call[0] for call in session.calls], ["post", "post", "get"])

    async def test_video_task_query_retries_rate_limit_response(self):
        session = _FakeSession(
            [
                ({"output": {"task_id": "task-1", "task_status": "PENDING"}}, 200),
                ({"code": "Throttling.RateQuota", "message": "Requests rate limit exceeded"}, 429),
                ({"output": {"task_status": "SUCCEEDED", "video_url": "https://example.com/video.mp4"}}, 200),
            ]
        )
        generator = VideoGeneratorDashScopeAPI(
            api_key="test-key",
            poll_interval=0,
            max_poll_attempts=1,
            rate_limit_retry_base_delay=0,
            rate_limit_retry_max_delay=0,
        )

        with patch("tools.video_generator_dashscope_api.aiohttp.ClientSession", return_value=session):
            output = await generator.generate_single_video("镜头缓慢推进", [])

        self.assertEqual(output.data, "https://example.com/video.mp4")
        self.assertEqual([call[0] for call in session.calls], ["post", "get", "get"])

    async def test_first_frame_video_uses_video_generation_endpoint_and_resolution(self):
        session = _FakeSession(
            [
                ({"output": {"task_id": "task-1", "task_status": "PENDING"}}, 200),
                ({"output": {"task_status": "SUCCEEDED", "video_url": "https://example.com/video.mp4"}}, 200),
            ]
        )
        generator = VideoGeneratorDashScopeAPI(api_key="test-key", poll_interval=0, max_poll_attempts=1)

        with tempfile.NamedTemporaryFile(suffix=".png") as frame:
            frame.write(b"first-frame")
            frame.flush()
            with patch("tools.video_generator_dashscope_api.aiohttp.ClientSession", return_value=session):
                await generator.generate_single_video("镜头缓慢推进", [frame.name], duration=4)

        create_call = session.calls[0]
        self.assertEqual(create_call[1], "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis")
        self.assertIn("img_url", create_call[2]["json"]["input"])
        self.assertEqual(create_call[2]["json"]["parameters"]["resolution"], "720P")
        self.assertEqual(create_call[2]["json"]["parameters"]["duration"], 4)
        self.assertNotIn("size", create_call[2]["json"]["parameters"])

    async def test_first_last_frame_video_uses_image2video_endpoint(self):
        session = _FakeSession(
            [
                ({"output": {"task_id": "task-1", "task_status": "PENDING"}}, 200),
                ({"output": {"task_status": "SUCCEEDED", "video_url": "https://example.com/video.mp4"}}, 200),
            ]
        )
        generator = VideoGeneratorDashScopeAPI(api_key="test-key", poll_interval=0, max_poll_attempts=1)

        with tempfile.NamedTemporaryFile(suffix=".png") as first, tempfile.NamedTemporaryFile(suffix=".png") as last:
            first.write(b"first-frame")
            first.flush()
            last.write(b"last-frame")
            last.flush()
            with patch("tools.video_generator_dashscope_api.aiohttp.ClientSession", return_value=session):
                await generator.generate_single_video("镜头缓慢推进", [first.name, last.name])

        create_call = session.calls[0]
        self.assertEqual(create_call[1], "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2video/video-synthesis")
        self.assertEqual(create_call[2]["json"]["input"].keys(), {"prompt", "first_frame_url", "last_frame_url"})
        self.assertEqual(create_call[2]["json"]["parameters"]["resolution"], "720P")
        self.assertNotIn("size", create_call[2]["json"]["parameters"])

    def test_video_payload_prefers_saved_source_url_sidecar(self):
        generator = VideoGeneratorDashScopeAPI(api_key="test-key")

        with tempfile.NamedTemporaryFile(suffix=".png") as frame:
            frame.write(b"image-bytes")
            frame.flush()
            with open(f"{frame.name}.source_url", "w", encoding="utf-8") as handle:
                handle.write("https://example.com/source.png\n")

            payload = generator._build_payload("镜头缓慢推进", [frame.name], "16:9", 5)

        self.assertEqual(payload["input"]["img_url"], "https://example.com/source.png")

    def test_image_url_save_writes_source_url_sidecar(self):
        image = ImageOutput(fmt="url", ext="png", data="https://example.com/image.png")

        with tempfile.NamedTemporaryFile(suffix=".png") as target:
            with patch("interfaces.image_output.download_image") as download_image:
                image.save(target.name)

            download_image.assert_called_once_with("https://example.com/image.png", target.name)
            with open(f"{target.name}.source_url", "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "https://example.com/image.png")

    async def test_failed_video_task_raises_provider_message(self):
        session = _FakeSession(
            [
                ({"output": {"task_id": "task-1", "task_status": "PENDING"}}, 200),
                ({"output": {"task_status": "FAILED", "message": "quota exceeded"}}, 200),
            ]
        )
        generator = VideoGeneratorDashScopeAPI(api_key="test-key", poll_interval=0, max_poll_attempts=1)

        with patch("tools.video_generator_dashscope_api.aiohttp.ClientSession", return_value=session):
            with self.assertRaisesRegex(RuntimeError, "quota exceeded"):
                await generator.generate_single_video("镜头缓慢推进", [])


if __name__ == "__main__":
    unittest.main()
