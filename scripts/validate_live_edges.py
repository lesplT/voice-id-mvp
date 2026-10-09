"""Read-only API checks: no enrollment records are changed."""
import io
import json
import wave
from pathlib import Path
from datetime import datetime, timezone
import httpx

def silence(seconds):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(bytes(int(seconds * 32000)))
    return b.getvalue()

results = []
for label, port, route, audio, status, reason in [
    ("short", 8002, "identify", silence(.5), 200, "too_short"),
    ("silence", 8002, "identify", silence(5), 200, "no_speech"),
    ("silent_dialogue", 8003, "dialogue", silence(5), 200, "no_speech"),
    ("invalid_audio", 8003, "transcribe", b"invalid", 400, None),
]:
    r = httpx.post(f"http://127.0.0.1:{port}/{route}",
                  files={"audio": ("edge.wav", audio, "audio/wav")}, timeout=120)
    body = r.json()
    passed = r.status_code == status and (reason is None or body.get("reason") == reason)
    results.append(dict(case=label, status=r.status_code, body=body, passed=passed))
report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), results=results)
path = Path(__file__).resolve().parents[1] / "data/edge-results.json"
path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
raise SystemExit(0 if all(r["passed"] for r in results) else 1)
