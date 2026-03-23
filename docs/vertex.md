# Vertex AI Support In This Fork

This fork adds a practical Vertex AI path for MineContext when you want Gemini models on Google Cloud instead of Doubao or a standard OpenAI-compatible provider.

## Scope

This fork changes three things:

- Vertex chat uses the OpenAI-compatible `chat/completions` endpoint with Google ADC-backed OAuth token refresh.
- Vertex embeddings use the native Vertex `publishers/google/models/*:predict` API.
- Desktop app validation accepts Vertex-style configuration without requiring a long-lived static API key.

## Why This Exists

The stock MineContext `custom` path had two problems for Vertex use:

1. Non-Doubao custom embeddings were routed through the wrong client path in MineContext.
2. Vertex chat and embeddings do not behave the same way:
   - chat works on the OpenAI-compatible endpoint
   - embeddings were not reliable on the OpenAI-compatible `/embeddings` endpoint in direct testing

For `gemini-embedding-001`, Google documents the native Vertex predict API. This fork uses that API directly for embeddings.

## Runtime Model Routing

When `base_url` looks like a Vertex OpenAI-compatible endpoint:

- chat requests stay on:
  - `https://aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/global/endpoints/openapi`
- embedding requests are rewritten to:
  - `https://us-central1-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/us-central1/publishers/google/models/gemini-embedding-001:predict`

The regional fallback is intentional. In direct testing for this fork:

- Vertex chat succeeded on `global`
- Vertex embedding succeeded on the native `us-central1` predict endpoint

## Authentication Model

This fork prefers Google Application Default Credentials.

Supported flows:

- `gcloud auth application-default login`
- `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json`

MineContext still has API key fields in the UI, but for Vertex they can be placeholders. The backend refreshes OAuth access tokens via ADC.

## MineContext Settings

Use `Custom` in the MineContext UI.

Vision language model:

- Model: `gemini-3-flash-preview`
- Base URL: `https://aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/global/endpoints/openapi`
- API key: `vertex-adc`

Embedding model:

- Model: `gemini-embedding-001`
- Base URL: `https://aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/global/endpoints/openapi`
- API key: `vertex-adc`

Even though the UI uses the same base URL for chat and embeddings, the backend rewrites Vertex embeddings to the native predict endpoint.

## Main Files

- `opencontext/llm/vertex_auth.py`
  - Vertex endpoint detection
  - ADC token refresh
  - native predict URL generation
- `opencontext/llm/llm_client.py`
  - chat remains OpenAI-compatible
  - Vertex embeddings use native predict
  - validation updated for Vertex
- `opencontext/server/routes/settings.py`
  - backend allows Vertex configurations without a real static API key
- `frontend/src/renderer/src/pages/settings/settings.tsx`
  - frontend no longer treats Vertex API key as mandatory
- `scripts/run_vertex_backend.sh`
  - helper to run the patched backend from source
- `scripts/packaged_backend_wrapper.sh`
  - wrapper used to redirect the installed app backend to the patched source checkout

## Current Limitations

- This fork assumes Vertex embeddings should use the native predict API, not the OpenAI-compatible embeddings route.
- The installed desktop app only benefits from these changes if it uses the patched backend.
- The local desktop wrapper path is machine-specific and may need adjusting on another computer.

## Reproducing The Backend Locally

```bash
cd ~/src/MineContext-vertex
uv python install 3.10
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .

gcloud auth application-default login

PROJECT_ID=your-project-id \
VERTEX_LOCATION=global \
scripts/run_vertex_backend.sh
```

## Suggested Next Cleanup

- Replace the machine-specific desktop wrapper with a documented install script
- Add automated tests for Vertex chat and native embedding routing
- Split Vertex support into a focused upstream PR if you want to contribute it back
