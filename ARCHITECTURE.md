# Package layout and code contracts

```text
voice_id_mvp/
  core/          # Validated configuration and project-root paths
  audio/         # FFmpeg conversion, WAV operations, energy-based speech detection
  recognition/   # Pretrained ASR, embeddings, speaker decisions and turn boundaries
  application/   # Dialogue workflow, dictionary corrections and bounded jobs
  storage/       # Qdrant voice repository, SQLite dictionary and recording journal
  services/      # FastAPI entrypoints and HTTP upload/error adapters
  web/           # Browser interface (HTML, CSS, JavaScript)
```

Imports name the full module, for example:

```python
from voice_id_mvp.audio.processing import normalized_upload
from voice_id_mvp.recognition.identification import identify_wav
from voice_id_mvp.storage.voice_repository import VoiceRepository
```

Relative imports and wildcard imports are disallowed. Model imports may remain lazy
to avoid loading PyTorch on health checks and to break intentional adapter cycles.
Every production Python class/function has an English docstring with three sections:

```python
"""Purpose:
    Explain the operation's responsibility.

Ingress:
    Describe inputs and relevant configuration or state.

Egress:
    Describe returned values, persisted effects, and important failures.
"""
```

`core.config.ROOT` still points to the repository root, not the new `core` directory.
Model files, `.env`, static assets and persistent paths therefore keep their previous locations.
HTTP entrypoints remain `voice_id_mvp.services.<service>_app:app`; URLs and response formats are unchanged.
Old flat Python module paths were internal implementation details and are replaced rather than re-exported.

## One Compose file

`docker-compose.yml` is the single configuration for the full stack:

```powershell
docker compose up -d --build --wait --wait-timeout 1800
```

The former small Compose file started only a standalone Qdrant for native Windows processes;
the former `compose.docker.yml` started the entire application. These configurations are now
merged. Native Windows launchers explicitly target `qdrant-native` under the optional `native`
profile, while normal Docker startup uses the private `qdrant` service.
Stopping native Windows services never stops the full stack's Qdrant.

The project name `voice-id-mvp-docker`, default UI port 18000, service names and `data/docker`
bind mounts are preserved. No journal, dictionary, voice-vector or model migration is needed.
