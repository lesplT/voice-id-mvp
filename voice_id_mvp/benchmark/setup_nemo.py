"""Install verified NVIDIA NeMo runtime and optional model files locally."""

import argparse
import hashlib
import platform
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from voice_id_mvp.core.config import ROOT
from voice_id_mvp.recognition.nemotron import MODEL_ARTIFACTS, RUNTIME_VERSION


_PACKAGES = {
    ("Windows", "amd64"): ("windows-x86_64-cpu.zip", "4ed9740e361e4653f56c3dce7d1e81883a1c8b831c1bd7d01567fff1130b5b05"),
    ("Linux", "x86_64"): ("linux-x86_64-cpu.tar.gz", "255f12b722ab15b2a38a51cf125d56e3fa97058b528f7e895b39d6a473deb06e"),
    ("Linux", "aarch64"): ("linux-aarch64-cpu.tar.gz", "56858eaf50d19b05902c043f65d0057352bdc107ae380c41053bfbddfc204bf1"),
}


def download_verified(url: str, destination: Path, digest: str) -> None:
    """Purpose: Retrieve an NVIDIA artifact and verify its pinned SHA-256 before use.
    Ingress: URL, destination path, and expected hash.
    Egress: A complete verified local file.
    """
    if destination.is_file():
        with destination.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() == digest:
                return
        raise RuntimeError(f"Unexpected checksum for existing file: {destination}")
    print(f"Downloading {destination.name}", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        checksum = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
                checksum.update(chunk)
        if checksum.hexdigest() != digest:
            raise RuntimeError(f"Unexpected checksum for {destination.name}")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def install_runtime() -> Path:
    """Purpose: Install the pinned official CPU build for this machine.
    Ingress: Host operating system and architecture.
    Egress: Path to the NeMo-Speech.cpp executable.
    """
    machine = platform.machine().lower()
    if platform.system() == "Windows" and machine == "x86_64":
        machine = "amd64"
    package = _PACKAGES.get((platform.system(), machine))
    if package is None:
        raise RuntimeError("Automatic NeMo setup supports Windows x64 and Linux x64/ARM64.")
    archive_name, checksum = package
    archive_name = f"nemo-speech-{RUNTIME_VERSION}-{archive_name}"
    folder_name = archive_name.removesuffix(".tar.gz").removesuffix(".zip")
    tools = ROOT / "models/nemo-tools"
    executable = tools / folder_name / "bin" / ("nemo-speech.exe" if platform.system() == "Windows" else "nemo-speech")
    if executable.is_file():
        return executable
    tools.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tools) as temp:
        stage = Path(temp)
        archive = stage / archive_name
        download_verified(f"https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v{RUNTIME_VERSION}/{archive_name}",
                          archive, checksum)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zipped:
                zipped.extractall(stage)
        else:
            with tarfile.open(archive) as packed:
                packed.extractall(stage, filter="data")
        shutil.move(str(stage / folder_name), tools / folder_name)
    return executable


def main() -> None:
    """Purpose: Install the NeMo runtime and the selected model weights.
    Ingress: The task requested by the command-line caller.
    Egress: Verified local artifacts and their printed paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", choices=["all", "asr", "diarization", "runtime"], default="all")
    args = parser.parse_args()
    print(f"NeMo runtime: {install_runtime()}")
    tasks = list(MODEL_ARTIFACTS) if args.task == "all" else ([] if args.task == "runtime" else [args.task])
    for task in tasks:
        repo, revision, filename, digest = MODEL_ARTIFACTS[task]
        path = ROOT / "models/nemo" / filename
        download_verified(f"https://huggingface.co/{repo}/resolve/{revision}/{filename}", path, digest)
        print(f"Verified {task}: {path}")


if __name__ == "__main__":
    main()
