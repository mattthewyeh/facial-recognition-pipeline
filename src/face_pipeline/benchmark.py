"""Warm, repeated CPU inference timings; no camera capture or display in the measurement."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import time

import cv2 as cv

from face_pipeline.detector import DEFAULT_DETECTOR_MODEL, YuNetFaceDetector
from face_pipeline.embedder import DEFAULT_EMBEDDING_MODEL, SFaceEmbedder
from face_pipeline.matcher import FaceMatcher
from face_pipeline.measurement import environment, summarize_ms, write_report
from face_pipeline.profiles import ProfileStore


STAGES = ("detection", "alignment", "embedding", "matching", "total")


def measure_frame(frame, detector, embedder, matcher=None, *, aligned=False, clock=time.perf_counter) -> dict:
    timings = {}
    start = clock()
    if aligned:
        crop = frame
    else:
        faces = detector.detect(frame)
        detection_end = clock()
        timings["detection"] = (detection_end - start) * 1000
        if len(faces) != 1:
            return {"status": "no_face" if not faces else "multiple_faces", "timings": {**timings, "total": (clock() - start) * 1000}}
        crop = embedder.align(frame, faces[0])
        alignment_end = clock()
        timings["alignment"] = (alignment_end - detection_end) * 1000
    embedding_start = clock()
    vector = embedder.extract_from_aligned(crop)
    embedding_end = clock()
    timings["embedding"] = (embedding_end - embedding_start) * 1000
    if matcher is not None:
        matcher.match(vector)
        timings["matching"] = (clock() - embedding_end) * 1000
    timings["total"] = (clock() - start) * 1000
    return {"status": "ok", "timings": timings}


def summarize_runs(runs: list[dict]) -> dict:
    successes = [r for r in runs if r["status"] == "ok"]
    return {"outcomes": dict(Counter(r["status"] for r in runs)),
            "all_attempts_total": summarize_ms([r["timings"]["total"] for r in runs]),
            "successful_attempts": {stage: summarize_ms([r["timings"][stage] for r in successes if stage in r["timings"]]) for stage in STAGES}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Recursively read local JPG/PNG inputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3, help="Warm-up passes over every image per configuration")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[0, 640], help="0 = original resolution")
    parser.add_argument("--aligned", action="store_true", help="112x112 aligned crops: embedding-only, or embedding + matching")
    parser.add_argument("--profiles-dir", type=Path, help="Optional existing local profile store for timing real matching")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--detector-model", type=Path, default=DEFAULT_DETECTOR_MODEL)
    parser.add_argument("--embedding-model", type=Path, default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()
    try:
        if args.repeats < 1 or args.warmup < 1 or args.threads < 1 or not args.dimensions or min(args.dimensions) < 0:
            raise ValueError("Repeats, warm-up and threads must be positive; dimensions must be nonnegative")
        paths = sorted(p for p in args.images.expanduser().rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not paths:
            raise ValueError("No images found")
        cv.setNumThreads(args.threads)
        dimensions = [0] if args.aligned else sorted(set(args.dimensions))
        shapes, digests = [], []
        for path in paths:
            frame = cv.imread(str(path))
            if frame is None:
                raise ValueError("An input image is unreadable")
            h, w = frame.shape[:2]
            if args.aligned and (h, w) != (112, 112):
                raise ValueError("Aligned mode requires 112x112 SFace crops")
            shapes.append((w, h))
            digests.append(hashlib.sha256(str(frame.shape).encode() + frame.tobytes()).hexdigest())
        max_dimension = max(max(shape) for shape in shapes)
        profiles = []
        matcher = None
        if args.profiles_dir is not None:
            if not args.profiles_dir.is_dir():
                raise ValueError("Profile directory does not exist")
            profiles = ProfileStore(args.profiles_dir).load_all()
            if not profiles:
                raise ValueError("Profile store is empty")
            matcher = FaceMatcher(profiles)
        embedder = SFaceEmbedder(args.embedding_model)
        detectors = {d: None if args.aligned else YuNetFaceDetector(args.detector_model, max_input_dimension=d or max_dimension) for d in dimensions}
        # Request settings after model loading; some builds ignore thread controls.
        cv.setNumThreads(args.threads)
        cv.ocl.setUseOpenCL(False)
        runs = {d: [] for d in dimensions}
        paired = {d: {} for d in dimensions}
        # Alternate configuration order each pass to reduce systematic warm/cache bias.
        for iteration in range(args.warmup + args.repeats):
            order = dimensions if iteration % 2 == 0 else list(reversed(dimensions))
            for image_index, path in enumerate(paths):
                frame = cv.imread(str(path))  # I/O deliberately excluded from stage timings.
                if frame is None:
                    raise ValueError("An input image became unreadable")
                for d in order:
                    result = measure_frame(frame, detectors[d], embedder, matcher, aligned=args.aligned)
                    if iteration >= args.warmup:
                        runs[d].append(result)
                        paired[d][(iteration, image_index)] = result
        report = {"schema_version": 1, "environment": environment(args.detector_model, args.embedding_model),
                  "scope": "aligned embeddings" if args.aligned else "single-face detection, alignment and embeddings",
                  "matching_included": matcher is not None, "gallery_identities": len(profiles),
                  "images": len(paths), "unique_images": len(set(digests)), "input_sizes": sorted(set(shapes)),
                  "dataset_sha256": hashlib.sha256("".join(sorted(digests)).encode()).hexdigest(),
                  "warmup_passes": args.warmup, "measured_passes": args.repeats, "requested_threads": args.threads,
                  "excluded": ["image decoding", "model loading", "camera capture", "display", "enrollment"],
                  "configurations": {str(d): summarize_runs(runs[d]) for d in dimensions},
                  "comparisons": {},
                  "warnings": ([f"Requested {args.threads} OpenCV threads, but this build reports {cv.getNumThreads()}; do not claim the requested thread count."] if cv.getNumThreads() != args.threads else []),
                  "limitations": "Warm CPU timing on repeated inputs, not live webcam FPS or recognition accuracy. Failed acquisitions are reported separately. Matching known enrollment crops is not an accuracy evaluation."}
        if not args.aligned and 0 in paired:
            for d in dimensions:
                if not d:
                    continue
                keys = [key for key in paired[0] if paired[0][key]["status"] == paired[d][key]["status"] == "ok"]
                baseline = [paired[0][key]["timings"]["total"] for key in keys]
                resized = [paired[d][key]["timings"]["total"] for key in keys]
                b, r = summarize_ms(baseline), summarize_ms(resized)
                report["comparisons"][str(d)] = {"paired_successes": len(keys),
                    "images_actually_resized": sum(max(shape) > d for shape in shapes),
                    "full_median_ms": b["median_ms"], "resized_median_ms": r["median_ms"],
                    "median_speedup": b["median_ms"] / r["median_ms"] if r["median_ms"] else None}
        write_report(args.output, report)
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
        for d, result in report["configurations"].items():
            print(f"dimension={d} outcomes={result['outcomes']} successful total={result['successful_attempts']['total']}")
    except (ValueError, OSError, cv.error) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
