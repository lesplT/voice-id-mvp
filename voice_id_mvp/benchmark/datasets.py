"""Prepare small, reproducible subsets of Open STT and ICSI."""

import argparse
import csv
import hashlib
import json
import math
import re
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from shutil import copyfileobj

from voice_id_mvp.audio.processing import normalize_audio, wav_duration, write_wav_slice


OPEN_STT = "https://azureopendatastorage.blob.core.windows.net/openstt"
ICSI = "https://huggingface.co/datasets/ggfox00000/dia-ICSIMeetingCorpus-all"
ICSI_REVISION = "e0cf8ffbd8d4e6a876378e68bdba0119b382c9c8"


def download(url: str, path: Path) -> Path:
    """Purpose: Cache a complete public dataset file atomically.
    Ingress: Source URL and local filename.
    Egress: Path to the cached file, or a network error.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=90) as source, partial.open("wb") as target:
                copyfileobj(source, target, 1024 * 1024)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    return path


def prepare_calls(root: Path, count: int, seed: str) -> None:
    """Purpose: Sample human-transcribed Russian calls independent of model outputs.
    Ingress: Output root, number of clips, and fixed selection seed.
    Egress: Paired audio and plain-text references plus selection metadata.
    """
    manifest = download(f"{OPEN_STT}/ru_open_stt_opus/manifests/asr_calls_2_val.csv", root / "calls.csv")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = [(a, t) for a, t, _ in csv.reader(stream)
                if re.fullmatch(r"asr_calls_2_val/[0-9a-f]/[0-9a-f]{2}/[0-9a-f]+\.wav", a)
                and t == str(PurePosixPath(a).with_suffix(".txt"))]
    selected = sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{row[0]}".encode()).digest())[:count]

    def fetch(row):
        """Purpose: Download one sample's audio and corresponding transcript.
        Ingress: A vetted pair of file paths from the dataset manifest.
        Egress: Two matching files in the output audio folder.
        """
        audio, text = row
        for name in (audio, text):
            download(f"{OPEN_STT}/ru_open_stt_opus_unpacked/{name}", root / "audio" / Path(name).name)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, selected))
    (root / "selection.json").write_text(json.dumps({"subset": "asr_calls_2_val", "seed": seed,
        "sources": [a for a, _ in selected]}, indent=2), encoding="utf-8")


def prepare_icsi(root: Path, meetings: list[str], start: float, duration: float) -> None:
    """Purpose: Build annotated evaluation clips and disjoint earlier enrollment regions.
    Ingress: Meeting IDs and a fixed evaluation range within each interaction recording.
    Egress: WAVs, aligned RTTMs, and a manifest of eligible enrollment speech.
    """
    listing = download(f"{ICSI.replace('/datasets/', '/api/datasets/')}/tree/{ICSI_REVISION}/ICSI/Segments?limit=1000",
                       root / "sources" / "segments.json")
    files = [row["path"] for row in json.loads(listing.read_text(encoding="utf-8"))]
    for meeting in meetings:
        if not re.fullmatch(r"B[a-z]{2}\d{3}", meeting):
            raise ValueError(f"Invalid ICSI meeting ID: {meeting}")
        paths = [path for path in files if PurePosixPath(path).name.startswith(meeting + ".") and path.endswith(".segs.xml")]
        if not paths:
            raise ValueError(f"No ICSI NXT segment annotations for {meeting}")
        reference = set()
        for path in paths:
            source = download(f"{ICSI}/resolve/{ICSI_REVISION}/{path}", root / "sources" / PurePosixPath(path).name)
            for element in ET.parse(source).iter("segment"):
                try:
                    a, b = float(element.attrib["starttime"]), float(element.attrib["endtime"])
                    speaker = element.attrib["participant"]
                except (KeyError, ValueError):
                    continue
                if 0 <= a < b and math.isfinite(b) and speaker:
                    reference.add((a, b, speaker))
        source = download(f"{ICSI}/resolve/{ICSI_REVISION}/Signals/{meeting}/{meeting}.interaction.wav",
                          root / "sources" / f"{meeting}.wav")
        audio_dir, enrollment_dir = root / "audio", root / "enrollment"
        audio_dir.mkdir(parents=True, exist_ok=True)
        enrollment_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            wav = normalize_audio(source, Path(temp) / "meeting.wav")
            end = min(start + duration, wav_duration(wav))
            if end <= start:
                raise ValueError(f"Empty evaluation range for {meeting}.")
            key = f"{meeting}_{start:g}_{end:g}"
            write_wav_slice(wav, audio_dir / f"{key}.wav", start, end)
            (audio_dir / f"{key}.rttm").write_text("".join(
                f"SPEAKER {key} 1 {max(a,start)-start:.3f} {min(b,end)-max(a,start):.3f} <NA> <NA> {speaker} <NA> <NA>\n"
                for a, b, speaker in sorted(reference) if a < end and b > start), encoding="utf-8")
            if start > 0:
                write_wav_slice(wav, enrollment_dir / f"{key}.wav", 0, start)
                (enrollment_dir / f"{key}.json").write_text(json.dumps([
                    (a, min(b, start), speaker) for a, b, speaker in sorted(reference) if a < start
                ]), encoding="utf-8")
        (audio_dir / f"{key}.source.json").write_text(json.dumps({
            "meeting": meeting, "start": start, "end": end, "revision": ICSI_REVISION,
            "source": f"Signals/{meeting}/{meeting}.interaction.wav",
        }, indent=2), encoding="utf-8")


def main() -> None:
    """Purpose: Expose reproducible dataset preparation as a command.
    Ingress: Dataset selection, destination and sampling parameters from the CLI.
    Egress: Audio and references laid out for the folder runner.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="dataset", required=True)
    calls = sub.add_parser("open-stt")
    calls.add_argument("output", type=Path)
    calls.add_argument("--count", type=int, default=100)
    calls.add_argument("--seed", default="voice-id-mvp-v1")
    icsi = sub.add_parser("icsi")
    icsi.add_argument("output", type=Path)
    icsi.add_argument("--meeting", nargs="+", default=["Bdb001"])
    icsi.add_argument("--start", type=float, default=300)
    icsi.add_argument("--duration", type=float, default=120)
    args = parser.parse_args()
    if args.dataset == "open-stt":
        if args.count < 1:
            parser.error("--count must be positive")
        prepare_calls(args.output, args.count, args.seed)
    else:
        if not math.isfinite(args.start) or not math.isfinite(args.duration) or args.start < 0 or args.duration <= 0:
            parser.error("Use a nonnegative --start and a positive --duration")
        prepare_icsi(args.output, args.meeting, args.start, args.duration)


if __name__ == "__main__":
    main()
