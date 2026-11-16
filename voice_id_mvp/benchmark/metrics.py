"""Reference-based scoring and descriptive statistics for local recordings."""

import re
import unicodedata
from collections import Counter, defaultdict
from math import isfinite


Turn = tuple[float, float, str]


def tokens(text: str) -> list[str]:
    """Purpose: Apply one consistent orthographic normalization to every ASR model.
    Ingress: A transcript or a human-written reference.
    Egress: Case-folded words with the Russian yo/ye distinction removed and punctuation treated as separators.
    """
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold().replace("ё", "е"))


def word_errors(reference: str, hypothesis: str) -> dict:
    """Purpose: Score the minimum word edits against a reference.
    Ingress: Human reference and model transcript before normalization.
    Egress: Substitution, deletion, insertion, reference totals and WER ratio.
    """
    ref, hyp = tokens(reference), tokens(hypothesis)
    row = [(i, 0, 0, i) for i in range(len(hyp) + 1)]  # cost, S, D, I
    for i, expected in enumerate(ref, start=1):
        previous, row = row, [(i, 0, i, 0)]
        for j, received in enumerate(hyp, start=1):
            if expected == received:
                row.append(previous[j - 1])
                continue
            sub, delete, insert = previous[j - 1], previous[j], row[-1]
            row.append(min([
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            ], key=lambda item: item[0]))
    edits, substitutions, deletions, insertions = row[-1]
    return {"wer": edits / len(ref) if ref else None, "reference_words": len(ref),
            "substitutions": substitutions, "deletions": deletions, "insertions": insertions}


def union_turns(turns: list[Turn], duration: float) -> list[Turn]:
    """Purpose: Clip and merge overlapping or adjacent time intervals for each label.
    Ingress: Raw labeled bounds and recording length.
    Egress: Valid intervals, with one active interval per label at any instant.
    """
    by_label: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for a, b, label in turns:
        if isfinite(a) and isfinite(b) and 0 <= max(0.0, a) < min(duration, b):
            by_label[label].append((max(0.0, a), min(duration, b)))
    clean = []
    for label, spans in by_label.items():
        merged = []
        for a, b in sorted(spans):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        clean.extend((a, b, label) for a, b in merged)
    return sorted(clean)


def activity(reference: list[Turn], hypothesis: list[Turn], duration: float, collar: float = 0.0):
    """Purpose: Sweep all times at which either recording's active speaker set changes.
    Ingress: Reference and predicted turns, recording length and full collar width.
    Egress: Elementary duration spans and active speaker sets outside excluded collars.
    """
    events = defaultdict(list)
    events[0.0], events[duration] = [], []
    for index, turns in enumerate((reference, hypothesis)):
        for a, b, label in union_turns(turns, duration):
            events[a].append((index, label, 1))
            events[b].append((index, label, -1))
            if not index and collar:
                for border in (a, b):
                    events[max(0.0, border - collar / 2)].append((2, "", 1))
                    events[min(duration, border + collar / 2)].append((2, "", -1))
    active = (set(), set())
    masked = 0
    boundaries = sorted(events)
    for a, b in zip(boundaries, boundaries[1:]):
        for index, label, direction in events[a]:
            if index == 2:
                masked += direction
            elif direction > 0:
                active[index].add(label)
            else:
                active[index].discard(label)
        if not masked:
            yield b - a, frozenset(active[0]), frozenset(active[1])


def diarization_errors(reference: list[Turn], hypothesis: list[Turn], duration: float,
                       collar: float = 0.0, skip_overlap: bool = False) -> dict:
    """Purpose: Score DER after optimally mapping anonymous predicted labels within a recording.
    Ingress: Reference and hypothesis turns, scoring region, and boundary/overlap policy.
    Egress: DER, missed, false alarm and speaker confusion seconds, and the label mapping.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    spans = [(dt, r, h) for dt, r, h in activity(reference, hypothesis, duration, collar)
             if not (skip_overlap and len(r) > 1)]
    refs = sorted(set().union(*(r for _, r, _ in spans)))
    hyps = sorted(set().union(*(h for _, _, h in spans)))
    matches = Counter()
    for dt, ref, hyp in spans:
        for r in ref:
            for h in hyp:
                matches[r, h] += dt
    counts = np.array([[matches[r, h] for h in hyps] for r in refs])
    mapping = {}
    if counts.size:
        i_rows, i_cols = linear_sum_assignment(counts, maximize=True)
        mapping = {hyps[j]: refs[i] for i, j in zip(i_rows, i_cols) if counts[i, j] > 0}
    missed = false_alarm = confusion = total = overlap = 0.0
    for dt, ref, hyp in spans:
        total += len(ref) * dt
        overlap += dt if len(ref) > 1 else 0
        missed += max(0, len(ref) - len(hyp)) * dt
        false_alarm += max(0, len(hyp) - len(ref)) * dt
        correct = sum(mapping.get(h) in ref for h in hyp)
        confusion += (min(len(ref), len(hyp)) - correct) * dt
    return {"der": (missed + false_alarm + confusion) / total if total else None,
            "missed_speaker_seconds": missed, "false_alarm_speaker_seconds": false_alarm,
            "confused_speaker_seconds": confusion, "reference_speaker_seconds": total,
            "reference_overlap_seconds": overlap, "mapping": mapping}


def identity_errors(reference: list[Turn], hypothesis: list[Turn], duration: float, enrolled: set[str]) -> dict:
    """Purpose: Score named Voice ID on clearly evaluable single-speaker regions.
    Ingress: Reference turns, literal speaker predictions and genuinely enrolled labels.
    Egress: Correct, wrong, abstained, and unknown-speaker false accepts in seconds.
    """
    counts = dict(enrolled_seconds=0.0, correct_seconds=0.0, wrong_seconds=0.0, unknown_seconds=0.0,
                  missed_seconds=0.0, unenrolled_seconds=0.0, false_accept_seconds=0.0)
    for dt, ref, hyp in activity(reference, hypothesis, duration):
        if len(ref) != 1:
            continue
        label = next(iter(ref))
        if label not in enrolled:
            counts["unenrolled_seconds"] += dt
            if hyp - {"UNKNOWN"}:
                counts["false_accept_seconds"] += dt
            continue
        counts["enrolled_seconds"] += dt
        key = ("missed" if not hyp else "unknown" if hyp == {"UNKNOWN"}
               else "correct" if hyp == {label} else "wrong")
        counts[f"{key}_seconds"] += dt
    return {**counts, "named_accuracy": counts["correct_seconds"] / counts["enrolled_seconds"]
            if counts["enrolled_seconds"] else None}


def turn_counts(turns: list[Turn], duration: float) -> dict:
    """Purpose: Describe diarization output even without reference annotations.
    Ingress: Predicted turns and audio duration.
    Egress: Turn count, label count, speech and overlap coverage in seconds.
    """
    clean = union_turns(turns, duration)
    spans = list(activity([], clean, duration))
    return {"segments": len(turns), "speaker_labels": len({s for _, _, s in clean if s != "UNKNOWN"}),
            "speech_seconds": sum(dt for dt, _, h in spans if h),
            "overlap_seconds": sum(dt for dt, _, h in spans if len(h) > 1),
            "unknown_seconds": sum(dt for dt, _, h in spans if "UNKNOWN" in h)}
