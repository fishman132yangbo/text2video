# text2video

text2video is a browser-based video generation console for turning ideas or scripts into generated videos. It provides a FastAPI backend, a static Web UI, configurable model providers, and task progress tracking for idea-to-video and script-to-video workflows.

For Chinese documentation, see [README_ZH.md](README_ZH.md).

## Features

- Idea-to-video and script-to-video generation modes.
- Browser console with live task status, event logs, and video download.
- Config-driven chat, image, and video generator backends.
- DashScope image and video provider integrations.
- Local runtime output under `.working_dir/web_jobs/`.

## Requirements

- Python 3.12+
- uv
- Provider credentials for the configured chat, image, and video models

## Quick Start

Install dependencies:

```bash
uv sync
```

Configure provider settings:

```bash
cp configs/idea2video.yaml configs/idea2video.local.yaml
cp configs/script2video.yaml configs/script2video.local.yaml
```

Edit the local config files with your provider API keys. Local config files are intended for private credentials and should not be committed.

Start the Web console:

```bash
./vimax web
```

Open the console:

```text
http://127.0.0.1:7860
```

To bind a different host or port:

```bash
./vimax web --host 0.0.0.0 --port 7860
```

## Configuration

The Web server loads local config files first:

- `configs/idea2video.local.yaml`
- `configs/script2video.local.yaml`

If a local file is missing, it falls back to:

- `configs/idea2video.yaml`
- `configs/script2video.yaml`

Each config defines:

- `chat_model`: script and planning model settings
- `image_generator`: image generation backend
- `video_generator`: video generation backend
- `working_dir`: workflow output directory

## Testing

Run the test suite:

```bash
./.venv/bin/python -m unittest discover tests
```

Or with uv:

```bash
uv run python -m unittest discover tests
```

## Repository Layout

```text
agents/       Generation roles for scripts, references, portraits, and shots
configs/      Runtime provider configuration templates
interfaces/   Shared data models
pipelines/    Idea-to-video and script-to-video workflows
tools/        Provider integrations and rendering backends
utils/        Shared helpers
web/          Static browser console
web_server.py FastAPI backend for the Web console
```

## Security

Do not commit real API keys. Keep credentials in ignored local config files or environment variables.
