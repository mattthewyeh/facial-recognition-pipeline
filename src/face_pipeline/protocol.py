"""Explicit, leakage-checked enrollment/validation/test manifests."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2 as cv


@dataclass(frozen=True)
class Sample:
    path: Path
    identity: str
    split: str
    session: str
    digest: str


def load_manifest(path: Path) -> list[Sample]:
    path = path.expanduser().resolve()
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a nonempty JSON array")
    samples = []
    seen_pixels: set[str] = set()
    sessions: dict[tuple[str, str], str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("path", "identity", "split", "session")
        ):
            raise ValueError(f"Row {index}: path, identity, split and session are required strings")
        if row["split"] not in {"enrollment", "validation", "test"}:
            raise ValueError(f"Row {index}: invalid split")
        image_path = (path.parent / row["path"]).resolve()
        frame = cv.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Row {index}: unreadable image")
        digest = hashlib.sha256(str(frame.shape).encode() + frame.tobytes()).hexdigest()
        if digest in seen_pixels:
            raise ValueError(f"Row {index}: duplicate image pixels; remove copied samples")
        seen_pixels.add(digest)
        key = (row["identity"], row["session"])
        if key in sessions and sessions[key] != row["split"]:
            raise ValueError(f"Row {index}: a capture session crosses splits")
        sessions[key] = row["split"]
        samples.append(Sample(image_path, row["identity"], row["split"], row["session"], digest))
    known = {s.identity for s in samples if s.split == "enrollment"}
    if not known:
        raise ValueError("Enrollment must contain at least one identity")
    unknowns = {}
    for split in ("validation", "test"):
        identities = {s.identity for s in samples if s.split == split}
        if not known <= identities:
            raise ValueError(f"{split}: include held-out images of every enrolled identity")
        unknowns[split] = identities - known
        if not unknowns[split]:
            raise ValueError(f"{split}: include at least one unenrolled identity")
    if unknowns["validation"] & unknowns["test"]:
        raise ValueError("Validation and test unknown identities must be disjoint")
    return samples


def dataset_summary(samples: list[Sample]) -> dict:
    # Fingerprint content and assignments, without writing paths or identity labels.
    assignments = [(s.digest, s.identity, s.split, s.session) for s in samples]
    fingerprint = hashlib.sha256(json.dumps(sorted(assignments)).encode()).hexdigest()
    return {
        "sha256": fingerprint,
        "splits": {
            split: {"images": sum(s.split == split for s in samples),
                    "identities": len({s.identity for s in samples if s.split == split})}
            for split in ("enrollment", "validation", "test")
        },
        "limitations": "Exact decoded duplicates and declared session overlap are rejected; near-duplicates and incorrect session labels still require manual review.",
    }
