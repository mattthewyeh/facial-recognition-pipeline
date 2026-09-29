# Reproducible performance measurements

Install the current CLI entry points with `python -m pip install -e ".[dev]"`.
Close heavy background applications, keep the laptop's power mode consistent and
record whether it is plugged in. Repeat the entire experiment to assess variability.
Hardware/software metadata and model SHA-256 hashes are included automatically.
No camera is opened; these are offline warm-inference measurements.

## Compare original frames with resized detection

```sh
benchmark-recognition --images data/evaluation --profiles-dir data/profiles --dimensions 0 640 --warmup 3 --repeats 20 --threads 1 --output results/benchmark-frames.json
```

Inputs must be original single-person JPG/PNG frames. JSON manifests are ignored.
Use the same set for both configurations; do not benchmark only the easier images
for one mode. `0` disables detector resizing by setting its limit to the largest
input dimension. Resizing affects detection only; alignment still uses the original
frame and restored landmarks. Profile matching is included only when
`--profiles-dir` is supplied; gallery size is recorded.

Every input is warmed up for every configuration. Measured passes alternate the
configuration order. Image decoding, enrollment, model loading, camera capture and
display are excluded. Models are loaded once and OpenCL is disabled for these CPU runs. Requested and
reported OpenCV thread counts are recorded. Some builds ignore `setNumThreads`;
a mismatch produces a warning and must not be reported as a single-thread run. Input size distribution, dataset fingerprint, unique image count and
sample count make conditions inspectable. Repeats are timing observations, not new
independent identities or test images.

Reports include mean, median and p95 milliseconds for detection, alignment,
embedding, matching and total processing. Stage summaries cover successful
single-face attempts; skipped stages have count zero and null times. Detection
failures are also summarized, with all-attempt total latency kept separately.
A faster run caused by missing faces must not be presented as an optimization win.

Speedup is the ratio of median total times on **paired successful attempts** for
the same input and pass. `images_actually_resized` must be nonzero before describing
a resizing benefit. Accuracy is not inferred from timing: use `evaluate-heldout`
on the fixed dataset to report recognition tradeoffs. These numbers are not live
webcam FPS, which also depends on capture, display and scheduling.

## Existing enrollment crops

The profile store contains already aligned 112x112 faces. For those inputs use:

```sh
benchmark-recognition --images data/profiles --aligned --profiles-dir data/profiles --warmup 3 --repeats 20 --threads 1 --output results/benchmark-aligned.json
```

This measures **embedding extraction plus matching only**. It does not measure
face detection, alignment or any original-frame resizing improvement. Such crops
are usable for timing but not independent recognition evaluation against their own
profiles. The report explicitly labels that limitation. In aligned mode dimension
comparisons are disabled.

All reports remain ignored under `results/`. Never commit private samples or
embeddings. Keep timing results with their metadata instead of copying an isolated
number into a résumé. Example wording after measurement: "Measured median/p95
embedding-and-matching latency of X/Y ms across N local crops and R repeated
passes on [hardware], with [reported thread count] OpenCV threads." Replace values only from an actual
saved report; no benchmark scores are shipped as universal claims.
