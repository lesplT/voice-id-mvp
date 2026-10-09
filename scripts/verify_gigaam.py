"""Verify the real CPU backend; no Vosk fallback is permitted in this check."""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voice_id_mvp.audio import normalize_audio
from voice_id_mvp.asr import GigaAmAsr

started = time.monotonic()
report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "engine": "gigaam"}
try:
    import torch
    import torchaudio
    torch.set_num_threads(4)
    report.update(torch=torch.__version__, torchaudio=torchaudio.__version__)
    source = ROOT / "data/synthetic_russian_tts.wav"
    target = ROOT / "data/gigaam-input.wav"
    normalize_audio(source, target)
    result = GigaAmAsr().transcribe(target)
    if not result["text"]:
        raise RuntimeError("Empty transcript")
    report.update(status="passed", result=result)
except Exception as exc:
    report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
finally:
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    (ROOT / "data/gigaam-results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
sys.exit(0 if report["status"] == "passed" else 1)
