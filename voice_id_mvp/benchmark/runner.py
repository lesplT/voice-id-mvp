"""Run recognition comparisons over an audio folder and save auditable reports."""

import argparse
import csv
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Callable

from voice_id_mvp.application.dialogue import legacy_timings, merge_dialogue_segments
from voice_id_mvp.audio.processing import normalize_audio, wav_duration, write_wav_slice
from voice_id_mvp.benchmark.metrics import activity, diarization_errors, identity_errors, tokens, turn_counts, word_errors
from voice_id_mvp.core.config import get_settings
from voice_id_mvp.recognition.asr import GigaAmAsr, VoskAsr
from voice_id_mvp.recognition.embeddings import get_embedder
from voice_id_mvp.recognition.identification import identify_wav
from voice_id_mvp.recognition.nemotron import RUNTIME_VERSION, diarize_nemotron, read_nemotron_transcript, transcribe_nemotron_directory


ASR = ("gigaam", "vosk", "nemotron-asr")
DIAR = ("legacy", "nemotron-diar")
FORMATS = {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac", ".aac", ".mp4", ".webm", ".wma", ".aiff", ".aif"}


class BenchmarkRoster:
    """Purpose: Emulate reference-only cosine search without touching the live voice database.
    Ingress: Per-speaker enrollment embeddings.
    Egress: Production-compatible candidate hits and reference vectors.
    """

    def __init__(self, vectors=None):
        """Purpose: Create a read-only view of references for one run.
        Ingress: An optional dictionary of embeddings by speaker.
        Egress: A roster used during baseline recognition.
        """
        self.vectors = vectors or {}

    def reference_vectors(self):
        """Purpose: Expose reference vectors to the speaker-change detector.
        Ingress: The roster stored in this instance.
        Egress: The per-speaker reference mapping.
        """
        return self.vectors

    def search(self, vector, limit):
        """Purpose: Return the closest normalized reference vectors.
        Ingress: A normalized query and a maximum result count.
        Egress: Reference hits sorted by descending cosine similarity.
        """
        import numpy as np

        hits = [{"speaker_id": label, "kind": "reference", "score": float(np.dot(vector, registered))}
                for label, refs in self.vectors.items() for registered in refs]
        return sorted(hits, key=lambda hit: hit["score"], reverse=True)[:limit]


def build_roster(root: Path, audio_name: str, work: Path) -> BenchmarkRoster:
    """Purpose: Enroll voices from clean, disjoint ICSI prefix regions.
    Ingress: Prefix WAV and annotations sharing the evaluation recording's stem.
    Egress: Up to three reference vectors per speaker from distinct single-speaker intervals.
    """
    wav = root / Path(audio_name).with_suffix(".wav")
    notes = wav.with_suffix(".json")
    if not wav.is_file() or not notes.is_file():
        raise FileNotFoundError(f"Missing enrollment WAV/JSON for {audio_name}")
    reference = json.loads(notes.read_text(encoding="utf-8"))
    candidates = defaultdict(list)
    cursor = 0.0
    for length, ref, _ in activity(reference, [], wav_duration(wav)):
        if len(ref) == 1 and length >= 1.5:
            candidates[next(iter(ref))].append((length, cursor, cursor + length))
        cursor += length
    vectors = {}
    clip = work / "enroll.wav"
    embedder = get_embedder()
    for label, spans in candidates.items():
        for _, start, end in sorted(spans, key=lambda s: (-s[0], s[1]))[:3]:
            write_wav_slice(wav, clip, start, min(start + 5, end))
            samples = embedder.extract_all(clip)
            if samples:
                vectors.setdefault(label, []).extend(samples)
    return BenchmarkRoster(vectors)


def read_rttm(path: Path) -> list[tuple[float, float, str]]:
    """Purpose: Load one annotated file without silently mixing recording IDs.
    Ingress: A sidecar RTTM describing one WAV.
    Egress: Timestamped speaker turns or a clear input error.
    """
    turns, recordings = [], set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] != "SPEAKER":
            continue
        if len(fields) < 8:
            raise ValueError(f"Invalid RTTM line in {path.name}")
        records = fields[1]
        recordings.add(records)
        start, length = float(fields[3]), float(fields[4])
        turns.append((start, start + length, fields[7]))
    if len(recordings) > 1:
        raise ValueError(f"Expected one recording ID in {path.name}")
    return turns


def legacy_diarization(wav: Path, work: Path, roster: BenchmarkRoster) -> list[tuple[float, float, str]]:
    """Purpose: Measure the same diarization decisions used in the original dialogue path.
    Ingress: Normalized recording, temporary directory and enrollment roster.
    Egress: VAD and speaker-change intervals labeled by existing Voice ID, including UNKNOWN.
    """
    result = []
    for part in legacy_timings(wav, roster):
        clip = work / "legacy-segment.wav"
        write_wav_slice(wav, clip, part.start, part.end)
        identity = identify_wav(clip, repository=roster)
        result.append({"start": round(part.start, 3), "end": round(part.end, 3), "speaker_id": identity["speaker_id"], "text": ""})
    return [(turn["start"], turn["end"], turn["speaker_id"]) for turn in merge_dialogue_segments(result)]


def write_json(path: Path, value) -> None:
    """Purpose: Save a JSON result atomically.
    Ingress: A target path and serializable value.
    Egress: A complete UTF-8 result file.
    """
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)


def _summary(rows: list[dict], names: tuple[str, ...]) -> dict:
    """Purpose: Aggregate quality only over recordings scored by all selected models in each group.
    Ingress: Per-recording outcomes and the selected models.
    Egress: Pooled WER/DER, coverage and component totals by model.
    """
    summary = {}
    for group, models, key, denominator, components in (
        ("asr", ASR, "wer", "reference_words", ("substitutions", "deletions", "insertions")),
        ("diarization", DIAR, "der", "reference_speaker_seconds",
         ("missed_speaker_seconds", "false_alarm_speaker_seconds", "confused_speaker_seconds")),
    ):
        active = [name for name in models if name in names]
        if not active:
            continue
        common = [row for row in rows if all(row["models"].get(name, {}).get("metrics", {}).get(key) is not None
                                            for name in active)]
        summary[group] = {}
        for name in active:
            valid = [row["models"][name] for row in rows if row["models"].get(name, {}).get("status") == "ok"]
            measured = [row["models"][name]["metrics"] for row in common]
            totals = {metric: sum(item[metric] for item in measured) for metric in (denominator, *components)}
            extra = {"recognized_words": sum(item["words"] for item in valid)} if group == "asr" else {
                "segments": sum(item["segments"] for item in valid),
                "speech_seconds": sum(item["speech_seconds"] for item in valid),
                "overlap_seconds": sum(item["overlap_seconds"] for item in valid),
            }
            counts = [row["models"][name]["identity"] for row in common if "identity" in row["models"][name]]
            if counts:
                named = {field: sum(item[field] for item in counts) for field in counts[0] if field.endswith("_seconds")}
                extra["voice_id"] = {**named, "named_accuracy": named["correct_seconds"] / named["enrolled_seconds"]
                                     if named["enrolled_seconds"] else None}
            summary[group][name] = {
                "successful_files": len(valid), "total_files": len(rows), "common_scored_files": len(measured),
                key: sum(totals[field] for field in components) / totals[denominator] if totals[denominator] else None,
                **totals, **extra,
            }
    return summary


def _save_reports(output: Path, rows: list[dict], options: dict, model_names: tuple[str, ...]) -> dict:
    """Purpose: Save spreadsheet-friendly per-file results and a machine-readable summary.
    Ingress: Evaluated rows, run configuration and output directory.
    Egress: The comparison CSV and summary JSON plus the summary value.
    """
    summary = {"configuration": options, "files": len(rows), **_summary(rows, model_names)}
    write_json(output / "summary.json", summary)
    fields = ["file", "duration_seconds", "model", "status", "wall_seconds", "words", "wer",
              "reference_words", "substitutions", "deletions", "insertions", "segments", "speaker_labels",
              "speech_seconds", "overlap_seconds", "unknown_seconds", "der", "reference_speaker_seconds",
              "missed_speaker_seconds", "false_alarm_speaker_seconds", "confused_speaker_seconds",
              "named_accuracy", "error", "reference_error"]
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        table = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", delimiter=";")
        table.writeheader()
        for row in rows:
            for name, result in row["models"].items():
                entry = {**result, **result.get("metrics", {}), **result.get("identity", {}),
                    "reference_error": ("; ".join(row.get("reference_errors", {}).values()) or None),
                    "file": row["file"], "model": name, "duration_seconds": row.get("duration_seconds")}
                table.writerow({key: "'" + entry[key] if isinstance(entry.get(key), str) and
                                entry[key].startswith(("=", "+", "-", "@", "\t", "\n", "\r")) else entry.get(key)
                                for key in fields})
    return summary


def run_folder(source: Path, output: Path, model_names: tuple[str, ...] = (*ASR, *DIAR),
               *, enrollment: Path | None = None, use_enrolled: bool = False, collar: float = 0.0,
               skip_overlap: bool = False, progress: Callable[[str], None] = print) -> dict:
    """Purpose: Run selected models independently on the same folder contents.
    Ingress: Input/result paths, model choices, roster policy and reference scoring options.
    Egress: Per-file JSON, CSV and aggregate metrics, including model failures.
    """
    source, output = source.resolve(), output.resolve()
    if not source.is_dir() or source == output or (output.exists() and any(output.iterdir())):
        raise ValueError("Choose an existing audio folder and a separate, empty results folder.")
    files = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in FORMATS
                   and not path.is_relative_to(output))
    if not files:
        raise ValueError("No supported audio files were found.")
    output.mkdir(parents=True, exist_ok=True)
    notes = output / "recordings"
    notes.mkdir()
    rows = []
    asr = {"vosk": VoskAsr().transcribe, "gigaam": GigaAmAsr().transcribe}
    settings = get_settings()
    options = {"models": list(model_names), "der_collar_seconds": collar, "der_skip_overlap": skip_overlap,
               "enrollment": "disjoint annotated prefix" if enrollment else "live roster snapshot" if use_enrolled else "none",
               "gigaam_model": settings.gigaam_model_name, "speaker_engine": settings.speaker_engine,
               "nemotron_device": settings.nemo_device, "nemo_runtime": RUNTIME_VERSION}
    shared_roster = BenchmarkRoster()
    shared_roster_error = None
    if use_enrolled and "legacy" in model_names:
        from voice_id_mvp.storage.voice_repository import VoiceRepository
        live = settings
        if (settings.qdrant_local_path / "meta.json").is_file():
            live = settings.model_copy(update={"qdrant_url": "local"})
        try:
            if live.qdrant_url != "local" or (live.qdrant_local_path / "meta.json").is_file():
                shared_roster = BenchmarkRoster(VoiceRepository(live).reference_vectors(create_missing=False))
        except Exception as exc:
            shared_roster_error = str(exc)
    with tempfile.TemporaryDirectory(prefix="work-", dir=output) as temp:
        work = Path(temp)
        normalized = work / "normalized"
        normalized.mkdir()
        for index, path in enumerate(files, start=1):
            progress(f"Подготовка {index}/{len(files)}: {path.name}")
            row = {"file": path.relative_to(source).as_posix(), "models": {}}
            rows.append(row)
            wav = normalized / f"{index:06d}.wav"
            try:
                normalize_audio(path, wav)
                row["duration_seconds"] = wav_duration(wav)
                if row["duration_seconds"] <= 0:
                    raise ValueError("The audio is empty.")
            except Exception as exc:
                wav.unlink(missing_ok=True)
                row["models"] = {name: {"status": "error", "error": f"Audio: {exc}"} for name in model_names}
        nemo_error = None
        nemo_batch_seconds = None
        if "nemotron-asr" in model_names and any(normalized.iterdir()):
            progress(f"Nemotron ASR: {len(files)} файлов")
            start = perf_counter()
            try:
                transcribe_nemotron_directory(normalized, work / "nemo-text")
            except Exception as exc:
                nemo_error = str(exc)
            nemo_batch_seconds = perf_counter() - start
        options["nemotron_asr_batch_wall_seconds"] = nemo_batch_seconds
        for index, (path, row) in enumerate(zip(files, rows), start=1):
            wav = normalized / f"{index:06d}.wav"
            if wav.is_file():
                progress(f"Сравнение {index}/{len(files)}: {path.name}")
                _compare(path, row, index, wav, work, model_names, asr, nemo_error, shared_roster,
                         shared_roster_error, enrollment, collar, skip_overlap)
                wav.unlink()
            write_json(notes / f"{index:06d}.json", row)
        return _save_reports(output, rows, options, model_names)


def _compare(path, row, index, wav, work, names, asr, nemo_error, shared_roster, shared_roster_error,
             enrollment, collar, skip_overlap):
    """Purpose: Capture recognition and scoring results for a normalized recording.
    Ingress: One audio file, optional sibling references, selected models, and scratch paths.
    Egress: An updated per-file row, retaining predictions even if reference parsing fails.
    """
    refs = {}
    for key, suffix, load in (("asr", ".txt", lambda p: p.read_text(encoding="utf-8-sig")),
                              ("diar", ".rttm", read_rttm)):
        sibling = path.with_suffix(suffix)
        if sibling.is_file():
            try:
                refs[key] = load(sibling)
            except Exception as exc:
                row.setdefault("reference_errors", {})[key] = str(exc)
    metadata = path.with_suffix(".source.json")
    if metadata.is_file():
        try:
            row["source"] = json.loads(metadata.read_text(encoding="utf-8"))
        except ValueError as exc:
            row.setdefault("reference_errors", {})["source"] = str(exc)
    duration = row["duration_seconds"]
    roster = shared_roster
    roster_error = shared_roster_error
    if enrollment and "legacy" in names:
        try:
            roster = build_roster(enrollment, row["file"], work)
        except Exception as exc:
            roster_error = str(exc)
    for name in names:
        started = perf_counter()
        try:
            if name in ASR:
                if name == "nemotron-asr":
                    result = work / "nemo-text" / f"{index:06d}.json"
                    if not result.is_file():
                        raise RuntimeError(nemo_error or "NeMo produced no transcript for this file.")
                    text = read_nemotron_transcript(result)["text"]
                else:
                    text = asr[name](wav)["text"]
                data = {"text": text, "words": len(tokens(text))}
                if "asr" in refs:
                    data["metrics"] = word_errors(refs["asr"], text)
            else:
                if name == "legacy":
                    if roster_error:
                        raise RuntimeError(roster_error)
                    turns = legacy_diarization(wav, work, roster)
                else:
                    turns = [(interval.start, interval.end, label) for interval, label in diarize_nemotron(wav)]
                data = {**turn_counts(turns, duration), "turns": [
                    {"start": a, "end": b, "speaker": s} for a, b, s in turns]}
                if "diar" in refs:
                    data["metrics"] = diarization_errors(refs["diar"], turns, duration, collar, skip_overlap)
                    if name == "legacy" and (enrollment or roster.vectors):
                        data["enrolled_labels"] = sorted(roster.vectors)
                        data["identity"] = identity_errors(refs["diar"], turns, duration, set(roster.vectors))
            row["models"][name] = {"status": "ok", **data}
        except Exception as exc:
            row["models"][name] = {"status": "error", "error": str(exc)}
        if name != "nemotron-asr":
            row["models"][name]["wall_seconds"] = round(perf_counter() - started, 4)


def main() -> None:
    """Purpose: Run a folder comparison from PowerShell or another shell.
    Ingress: Input and output folders, models, and optional scoring settings.
    Egress: Saved reports and a nonzero exit status when a model fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--models", nargs="+", choices=(*ASR, *DIAR), default=[*ASR, *DIAR])
    roster = parser.add_mutually_exclusive_group()
    roster.add_argument("--enrollment", type=Path, help="Disjoint per-file enrollment material for ICSI")
    roster.add_argument("--use-enrolled", action="store_true", help="Read the local voice roster without changing it")
    parser.add_argument("--collar", type=float, default=0)
    parser.add_argument("--skip-overlap", action="store_true")
    args = parser.parse_args()
    from math import isfinite
    if not isfinite(args.collar) or args.collar < 0:
        parser.error("--collar must be finite and nonnegative")
    models = tuple(dict.fromkeys(args.models))
    summary = run_folder(args.source, args.output, models, enrollment=args.enrollment,
                         use_enrolled=args.use_enrolled, collar=args.collar, skip_overlap=args.skip_overlap)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if any(model["successful_files"] != summary["files"] for key in ("asr", "diarization")
           for model in summary.get(key, {}).values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
