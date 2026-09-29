"""Open-set identification metrics; validation selects, test only measures."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from face_pipeline.detector import DEFAULT_DETECTOR_MODEL, YuNetFaceDetector
from face_pipeline.embedder import DEFAULT_EMBEDDING_MODEL, SFaceEmbedder
from face_pipeline.matcher import DEFAULT_COSINE_THRESHOLD, FaceMatcher
from face_pipeline.profiles import LoadedProfile, Profile
from face_pipeline.protocol import Sample, dataset_summary, load_manifest
from face_pipeline.measurement import environment, write_report


@dataclass(frozen=True)
class Observation:
    identity: str
    predicted: str | None
    similarity: float | None
    status: str = "ok"


def gallery(samples: list[Sample], detector, embedder) -> list[LoadedProfile]:
    vectors: dict[str, list] = {}
    for sample in samples:
        if sample.split != "enrollment":
            continue
        frame = cv.imread(str(sample.path))
        if frame is None:
            raise ValueError("Enrollment image became unreadable")
        faces = detector.detect(frame)
        if len(faces) != 1:
            raise ValueError("Every enrollment image must have exactly one detected face")
        vectors.setdefault(sample.identity, []).append(embedder.extract(frame, faces[0]))
    return [LoadedProfile(Profile(identity, identity, "", "SFace", len(items), Path(".")),
                          np.stack(items)) for identity, items in sorted(vectors.items())]


def observe(samples: list[Sample], profiles, detector, embedder) -> dict[str, list[Observation]]:
    matcher = FaceMatcher(profiles, threshold=-1)
    result: dict[str, list[Observation]] = {"validation": [], "test": []}
    for sample in samples:
        if sample.split == "enrollment":
            continue
        frame = cv.imread(str(sample.path))
        if frame is None:
            raise ValueError("Evaluation image became unreadable")
        faces = detector.detect(frame)
        if len(faces) != 1:
            status = "no_face" if not faces else "multiple_faces"
            result[sample.split].append(Observation(sample.identity, None, None, status))
        else:
            match = matcher.match(embedder.extract(frame, faces[0]))
            result[sample.split].append(Observation(sample.identity, match.profile_id, match.similarity))
    return result


def metrics(rows: list[Observation], known: set[str], threshold: float) -> dict:
    if not np.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("Threshold must be finite and between -1 and 1")
    counts = dict(known_total=0, unknown_total=0, known_correct=0, known_false_rejects=0,
                  known_wrong_identity=0, unknown_false_accepts=0, unknown_correct_rejects=0,
                  known_acquisition_failures=0, unknown_acquisition_failures=0)
    for row in rows:
        kind = "known" if row.identity in known else "unknown"
        counts[kind + "_total"] += 1
        if row.status != "ok":
            counts[kind + "_acquisition_failures"] += 1
            continue
        if row.similarity is None or not np.isfinite(row.similarity) or row.predicted not in known:
            raise ValueError("Successful observations require a finite score and enrolled prediction")
        accepted = row.similarity >= threshold
        if kind == "known":
            key = "known_false_rejects" if not accepted else (
                "known_correct" if row.predicted == row.identity else "known_wrong_identity")
        else:
            key = "unknown_false_accepts" if accepted else "unknown_correct_rejects"
        counts[key] += 1
    k, u = counts["known_total"], counts["unknown_total"]
    if not k or not u:
        raise ValueError("Each evaluation split requires known and unknown probes")
    ku, uu = k - counts["known_acquisition_failures"], u - counts["unknown_acquisition_failures"]
    return {"threshold": threshold, **counts,
            "known_correct_rate": counts["known_correct"] / k,
            "unknown_correct_reject_rate": counts["unknown_correct_rejects"] / u,
            "balanced_success_rate": (counts["known_correct"] / k + counts["unknown_correct_rejects"] / u) / 2,
            "known_false_reject_rate_given_acquisition": counts["known_false_rejects"] / ku if ku else None,
            "unknown_false_accept_rate_given_acquisition": counts["unknown_false_accepts"] / uu if uu else None}


def select_threshold(rows: list[Observation], known: set[str]) -> tuple[float, list[dict]]:
    scores = [r.similarity for r in rows if r.status == "ok" and r.similarity is not None]
    for is_known in (True, False):
        if not any(r.status == "ok" and (r.identity in known) == is_known for r in rows):
            raise ValueError("Calibration needs successful known and unknown validation probes")
    candidates = {-1.0, 1.0, DEFAULT_COSINE_THRESHOLD}
    for score in scores:
        candidates.add(float(np.clip(score, -1, 1)))
        candidates.add(float(np.clip(np.nextafter(score, np.inf), -1, 1)))
    sweep = [metrics(rows, known, threshold) for threshold in sorted(candidates)]
    best = max(sweep, key=lambda m: (m["balanced_success_rate"], -m["unknown_false_accepts"], m["threshold"]))
    return best["threshold"], sweep


def build_report(observations: dict[str, list[Observation]], known: set[str]) -> dict:
    threshold, sweep = select_threshold(observations["validation"], known)
    return {"protocol": "open-set identification with enrollment centroids",
            "selection": "Maximize validation balanced success; ties prefer fewer unknown accepts, then higher threshold",
            "selected_threshold": threshold, "validation_sweep": sweep,
            "validation": metrics(observations["validation"], known, threshold),
            "test": metrics(observations["test"], known, threshold),
            "default_threshold_test": metrics(observations["test"], known, DEFAULT_COSINE_THRESHOLD),
            "limitations": "Image-level descriptive results, not population accuracy or fairness estimates. Repeated images of a person are correlated. Acquisition failures count as failures, not successful unknown rejection."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-input-dimension", type=int, default=640, help="0 = original resolution; default 640")
    parser.add_argument("--detector-model", type=Path, default=DEFAULT_DETECTOR_MODEL)
    parser.add_argument("--embedding-model", type=Path, default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()
    try:
        samples = load_manifest(args.manifest)
        if args.max_input_dimension < 0:
            raise ValueError("Maximum input dimension must be nonnegative")
        dimension = args.max_input_dimension or max(max(cv.imread(str(s.path)).shape[:2]) for s in samples)
        detector = YuNetFaceDetector(args.detector_model, max_input_dimension=dimension)
        embedder = SFaceEmbedder(args.embedding_model)
        profiles = gallery(samples, detector, embedder)
        report = build_report(observe(samples, profiles, detector, embedder), {p.profile.profile_id for p in profiles})
        report.update(dataset=dataset_summary(samples), environment=environment(args.detector_model, args.embedding_model),
                      detector_max_input_dimension=args.max_input_dimension, detector_score_threshold=0.8)
        write_report(args.output, report)
        print(json.dumps({"threshold": report["selected_threshold"], "test": report["test"]}, indent=2))
    except (ValueError, OSError, cv.error) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
