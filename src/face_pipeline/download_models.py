from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class ModelArtifact:
    filename: str
    url: str
    sha256: str


MODEL_ARTIFACTS = (
    ModelArtifact(
        filename="face_detection_yunet_2023mar.onnx",
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    ModelArtifact(
        filename="face_recognition_sface_2021dec.onnx",
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(artifact: ModelArtifact, model_dir: Path) -> Path:
    destination = model_dir / artifact.filename
    if destination.exists() and file_sha256(destination) == artifact.sha256:
        print(f"Verified existing model: {destination.name}")
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        artifact.url,
        headers={"User-Agent": "facial-recognition-pipeline/0.1"},
    )

    print(f"Downloading {artifact.filename}...")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("wb") as model_file:
                while chunk := response.read(1024 * 1024):
                    model_file.write(chunk)
    except (OSError, urllib.error.URLError):
        temporary.unlink(missing_ok=True)
        raise

    actual_hash = file_sha256(temporary)
    if actual_hash != artifact.sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Checksum mismatch for {artifact.filename}: expected "
            f"{artifact.sha256}, received {actual_hash}"
        )

    os.replace(temporary, destination)
    print(f"Saved and verified: {destination}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify the YuNet and SFace ONNX models."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Destination directory (default: {DEFAULT_MODEL_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        for artifact in MODEL_ARTIFACTS:
            download_model(artifact, model_dir)
    except (OSError, urllib.error.URLError, ValueError) as error:
        print(f"Model setup failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

