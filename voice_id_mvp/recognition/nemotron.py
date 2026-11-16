"""Run NVIDIA NeMo-Speech.cpp with pinned, local Nemotron GGUF models."""

import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from voice_id_mvp.audio.processing import wav_duration
from voice_id_mvp.audio.segmentation import TimeSegment
from voice_id_mvp.core.config import ROOT, get_settings


RUNTIME_VERSION = "0.3.0"
MODEL_ARTIFACTS = {
    "asr": (
        "nvidia/nemotron-3.5-asr-streaming-0.6b", "1c8deaecc64b91f034d73e08dd8b64625eb3395d",
        "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf", "a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae",
    ),
    "diarization": (
        "nvidia/Nemotron-3-Diarization", "f667ed73aee57d40cc39428eb768b4fd87a0a29e",
        "Nemotron-3-Diarization.q8_0.gguf", "08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1",
    ),
}


class NemotronUnavailable(RuntimeError):
    """Purpose: Report optional setup or inference failures.
    Ingress: Missing weights, missing executable, or failed runtime call.
    Egress: An actionable error without a silently substituted prediction.
    """


def nemo_binary() -> str:
    """Purpose: Locate the configured executable or the project-managed runtime.
    Ingress: Project settings and the runtime install directory.
    Egress: A command path, or a setup error.
    """
    configured = get_settings().nemo_speech_bin
    if configured == "nemo-speech":
        system = platform.system().lower()
        architecture = {"amd64": "x86_64", "arm64": "aarch64"}.get(platform.machine().lower(),
                                                                platform.machine().lower())
        filename = "nemo-speech.exe" if system == "windows" else "nemo-speech"
        local = ROOT / "models/nemo-tools" / f"nemo-speech-{RUNTIME_VERSION}-{system}-{architecture}-cpu" / "bin" / filename
        if local.is_file():
            return str(local)
    binary = shutil.which(configured)
    if not binary:
        raise NemotronUnavailable("NeMo is not installed. Run: python -m voice_id_mvp.benchmark.setup_nemo")
    return binary


def _command(task: str, path: Path, output: list[str]) -> subprocess.CompletedProcess:
    """Purpose: Invoke one official NeMo task with fixed model and decoding options.
    Ingress: Task, input WAV or folder, and arguments specifying the result path.
    Egress: Process diagnostics or an actionable exception.
    """
    binary = nemo_binary()
    weights = ROOT / "models/nemo" / MODEL_ARTIFACTS[task][2]
    if not weights.is_file():
        raise NemotronUnavailable(f"Nemotron {task} weights are missing. Run: "
                                  f"python -m voice_id_mvp.benchmark.setup_nemo {task}")
    if task == "asr":
        action = ["transcribe", "--language", "ru-RU", "--format", "json", "--verbatim"]
    else:
        action = ["diarize", "--preset", "v3-offline", "--format", "rttm"]
    try:
        result = subprocess.run(
            [binary, action[0], str(path.resolve()), *action[1:], "--model", str(weights),
             "--device", get_settings().nemo_device, *output],
            capture_output=True, encoding="utf-8", errors="replace", timeout=3600, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NemotronUnavailable(f"NeMo {task} could not finish: {exc}") from exc
    if result.returncode:
        raise NemotronUnavailable(f"NeMo {task} exited ({result.returncode}): {result.stderr.strip()[-1500:]}")
    return result


def diarize_nemotron(wav_path: Path) -> list[tuple[TimeSegment, str]]:
    """Purpose: Find anonymous speakers and preserve overlapping speech.
    Ingress: A normalized mono 16 kHz recording.
    Egress: Ordered (time interval, recording-local speaker label) pairs.
    """
    with tempfile.TemporaryDirectory(prefix="nemotron-diar-") as work:
        output = Path(work) / "output.rttm"
        _command("diarization", wav_path, ["--output", str(output)])
        lines = output.read_text(encoding="utf-8").splitlines()
    duration = wav_duration(wav_path)
    turns = []
    for line in lines:
        fields = line.split()
        if len(fields) < 8 or fields[0] != "SPEAKER":
            continue
        start, length = float(fields[3]), float(fields[4])
        end = min(duration, start + length)
        start = max(0.0, start)
        if start < end:
            turns.append((TimeSegment(start, end), fields[7]))
    return sorted(turns, key=lambda turn: (turn[0].start, turn[0].end, turn[1]))


def transcribe_nemotron(wav_path: Path) -> dict:
    """Purpose: Decode Russian text using the pinned Nemotron ASR checkpoint.
    Ingress: A normalized WAV recording.
    Egress: A transcription dictionary with the raw model text.
    """
    with tempfile.TemporaryDirectory(prefix="nemotron-asr-") as work:
        output = Path(work) / "output.json"
        _command("asr", wav_path, ["--output", str(output)])
        return read_nemotron_transcript(output)


def read_nemotron_transcript(output: Path) -> dict:
    """Purpose: Read the NeMo structured transcript format without guessing on failures.
    Ingress: JSON emitted by NeMo-Speech.cpp.
    Egress: Transcript, engine name, and available word positions.
    """
    raw = json.loads(output.read_text(encoding="utf-8"))
    return {"engine": "nemotron", "text": raw["text"].strip(), "words": raw.get("words") or []}


def transcribe_nemotron_directory(source: Path, destination: Path) -> None:
    """Purpose: Reuse a single NeMo load when decoding many clips.
    Ingress: A flat normalized WAV directory and an output directory for matching JSON files.
    Egress: One JSON result per decoded recording, or an explicit runtime error.
    """
    destination.mkdir(parents=True, exist_ok=True)
    _command("asr", source, ["--output-dir", str(destination.resolve()), "--concurrency", "1"])
