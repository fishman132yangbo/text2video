# Repository Guidelines

## Project Structure & Module Organization

text2video is a Python 3.12 browser video generation console. The FastAPI Web backend lives in `web_server.py`, video workflows in `pipelines/`, agent-like generation roles in `agents/`, shared data models in `interfaces/`, provider integrations in `tools/`, and helper code in `utils/`. The supported entry point is the Web console. Tests live in `tests/`. Web assets live in `web/`. Static assets are in `assets/`, and runtime/provider settings live in `configs/`.

## Build, Test, and Development Commands

- `uv sync`: create/update the Python environment from `pyproject.toml`.
- `uv run python -m unittest discover tests`: run the Python test suite.
- `uv run python web_server.py --port 7860`: run the FastAPI Web video console backend and static frontend.
- `./vimax web --port 7860`: start the browser video generation console.

## Coding Style & Naming Conventions

Use 4-space indentation for Python and keep modules snake_case. Classes use PascalCase; functions, variables, and test methods use snake_case. Keep provider classes named by capability and backend, for example `VideoGeneratorDashScopeAPI`. Web JavaScript uses camelCase and keeps DOM updates in small functions. Prefer small, explicit functions over broad utility abstractions.

## Testing Guidelines

Python tests use `unittest`; name files `tests/test_*.py` and classes around the behavior under test. Mock network/API calls and avoid requiring real model credentials in tests. Web API tests should use fake pipeline runners so provider calls are not triggered. Run the relevant Python tests before opening a PR.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, lowercase summaries. Keep commits focused. PRs should include a concise description, test results, linked issues when available, and screenshots or terminal output for Web/user-facing changes.

## Security & Configuration Tips

Do not commit real API keys. Use `configs/idea2video.yaml` and `configs/script2video.yaml` as templates. Keep runnable secrets in ignored local files such as `configs/idea2video.local.yaml` / `configs/script2video.local.yaml`, or in environment variables such as `VIMAX_LLM_API_KEY`, `VIMAX_IMAGE_API_KEY`, and `VIMAX_VIDEO_API_KEY`.

## Agent-Specific Instructions

When interacting with the local user, default to Chinese unless the user explicitly requests another language.
