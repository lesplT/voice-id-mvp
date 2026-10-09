from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "data/synthetic_russian_tts.wav"


def request(method: str, url: str, **kwargs):
    response = httpx.request(method, url, timeout=180, **kwargs)
    try:
        body = response.json()
    except Exception:
        body = response.text
    return {"status": response.status_code, "body": body}


def audio_file():
    return {"audio": (AUDIO.name, AUDIO.read_bytes(), "audio/wav")}


def main() -> int:
    if not AUDIO.exists():
        raise SystemExit(f"Missing synthetic audio: {AUDIO}")
    results = {
        "health": {
            str(port): request("GET", f"http://127.0.0.1:{port}/health")
            for port in (8001, 8002, 8003)
        }
    }
    request("DELETE", "http://127.0.0.1:8001/speakers/synthetic_tts")
    results["enroll"] = request(
        "POST",
        "http://127.0.0.1:8001/speakers/synthetic_tts/enroll",
        files=audio_file(),
    )
    results["speakers"] = request("GET", "http://127.0.0.1:8001/speakers")
    results["identify"] = request(
        "POST", "http://127.0.0.1:8002/identify", files=audio_file()
    )
    results["transcribe"] = request(
        "POST", "http://127.0.0.1:8003/transcribe", files=audio_file()
    )
    results["dialogue"] = request(
        "POST", "http://127.0.0.1:8003/dialogue", files=audio_file()
    )
    output = ROOT / "data/smoke-results.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    required = [*results["health"].values(), results["enroll"], results["identify"], results["transcribe"], results["dialogue"]]
    return 0 if all(item["status"] < 400 for item in required) else 1


if __name__ == "__main__":
    sys.exit(main())

