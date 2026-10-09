"""Validate current enrollment without changing or deleting it."""
import json
from pathlib import Path
from datetime import datetime, timezone
import httpx
root = Path(__file__).resolve().parents[1]
audio = root / "data/synthetic_russian_tts.wav"
report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "results": {}}
with httpx.Client(timeout=180) as client:
    for port in (8001, 8002, 8003):
        r = client.get(f"http://127.0.0.1:{port}/health")
        r.raise_for_status()
        report["results"][f"health_{port}"] = r.json()
    for port, endpoint in [(8002, "identify"), (8003, "transcribe"), (8003, "dialogue")]:
        r = client.post(f"http://127.0.0.1:{port}/{endpoint}",
                        files={"audio": (audio.name, audio.read_bytes(), "audio/wav")})
        r.raise_for_status()
        report["results"][endpoint] = r.json()
assert report["results"]["identify"]["speaker_id"] == "synthetic_tts"
assert report["results"]["transcribe"]["text"]
assert report["results"]["dialogue"]["segments"]
(root / "data/review-results.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
