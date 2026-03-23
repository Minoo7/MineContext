# PR Summary

## Title

Add Vertex ADC auth and native Vertex embedding support

## Problem

MineContext worked well for Doubao and normal OpenAI-compatible backends, but Vertex AI required different behavior:

- Vertex uses Google OAuth access tokens instead of long-lived API keys
- the MineContext `custom` embedding path was not suitable for Vertex
- `gemini-embedding-001` needed the native Vertex predict API in direct testing

## What Changed

### Backend

- added `opencontext/llm/vertex_auth.py`
  - detects Vertex-style base URLs
  - normalizes Gemini model names to `google/<model>` where needed
  - refreshes ADC access tokens automatically
  - builds native Vertex embedding predict URLs
- updated `opencontext/llm/llm_client.py`
  - keeps Vertex chat on OpenAI-compatible chat completions
  - routes Vertex embeddings to native `publishers/google/models/*:predict`
  - validates Vertex embeddings through the native path
- updated `opencontext/server/routes/settings.py`
  - allows Vertex-style configs without requiring a true static API key

### Frontend

- updated `frontend/src/renderer/src/pages/settings/settings.tsx`
  - stops forcing a real API key for Vertex URLs

### Supporting Fixes

- updated `opencontext/config/global_config.py`
  - improves config reload behavior for source-run backend flows
- updated `opencontext/utils/logging_utils.py`
  - avoids startup issues when config is not yet loaded
- updated `pyproject.toml`
  - adds dependencies needed by the Vertex path
- added scripts:
  - `scripts/run_vertex_backend.sh`
  - `scripts/packaged_backend_wrapper.sh`

## Behavior After This Change

For Vertex-style `base_url` values:

- chat requests use the OpenAI-compatible endpoint
- embedding requests use the native Vertex predict endpoint
- ADC refresh is used automatically

## Validation Performed

- direct chat call to Vertex OpenAI-compatible endpoint succeeded
- direct native predict call to `gemini-embedding-001` succeeded
- patched Python modules passed `py_compile`

## Notes

- The desktop wrapper is useful for local use but may be too machine-specific for upstream.
- The core backend Vertex changes are the main part worth upstreaming.
