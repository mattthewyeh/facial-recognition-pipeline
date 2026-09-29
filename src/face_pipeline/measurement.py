"""Report metadata and timing summaries without hostnames or personal paths."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import cv2 as cv
import numpy as np


def environment(detector_model: Path, embedding_model: Path) -> dict:
    cpu = platform.processor()
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return {"generated_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "opencv": cv.__version__, "numpy": np.__version__,
            "os": platform.system(), "os_release": platform.release(), "architecture": platform.machine(),
            "processor": cpu, "opencv_optimized": cv.useOptimized(), "opencl_enabled": cv.ocl.useOpenCL(), "opencv_threads": cv.getNumThreads(),
            "models_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in (("yunet", detector_model), ("sface", embedding_model))}}


def summarize_ms(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None}
    if any(not np.isfinite(v) or v < 0 for v in values):
        raise ValueError("Timing values must be finite and nonnegative")
    return {"count": len(values), "mean_ms": float(np.mean(values)),
            "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}


def write_report(path: Path, report: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
