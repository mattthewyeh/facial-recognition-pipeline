# Held-out recognition evaluation

This protocol evaluates **open-set identification**: enroll known people, identify
new photos of them, and reject people who were never enrolled. It is not the same
as comparing every possible pair of images. The original `evaluate-recognition`
command remains available for exploratory pairwise checks; its suggested threshold
and accuracy use the same dataset and are not held-out evidence.

## Prepare local data

Use consenting adults or a dataset licensed for this purpose. Keep photos, names,
embeddings, manifests and generated reports under ignored `data/` and `results/`.
The tool does not upload data. Reports contain aggregate counts and fingerprints,
not images, identity names, paths or embeddings. Reports still stay local by default.

Existing enrollment crops can serve as enrollment data, but cannot be repurposed
as independent test images. Collect **original camera frames** in distinct capture
sessions, not adjacent frames from the same video divided across splits.

A useful pilot is two or more enrolled people with several enrollment photos each,
plus validation and test sessions with changes in distance, pose and lighting.
Include at least one separate unknown person in validation and another in test.
For stronger evidence, expand the number of people and sessions substantially.
There is no magic sample count that establishes general reliability.

### Capture original webcam frames

The `capture-evaluation` command saves full, unannotated webcam frames and adds
them to `data/evaluation/manifest.json`. Press `Space` for each photo and vary
your angle, expression, distance and lighting. The command requires exactly one
detected face before accepting a frame.

Use a separate session label for every split. For example:

```sh
capture-evaluation --identity matthew --split enrollment --session matthew_day1 --samples 8
capture-evaluation --identity matthew --split validation --session matthew_day2 --samples 8
capture-evaluation --identity matthew --split test --session matthew_day3 --samples 8
```

Repeat those three commands for each enrolled person. Validation must also
contain a consenting person absent from enrollment. Test must contain a different
consenting person absent from enrollment. Those unknown participants only need
their respective validation or test capture command.

Create `data/evaluation/manifest.json` (paths relative to the manifest):

```json
[
  {"path": "enroll/a1.jpg", "identity": "person_a", "split": "enrollment", "session": "day1"},
  {"path": "enroll/b1.jpg", "identity": "person_b", "split": "enrollment", "session": "day1"},
  {"path": "validation/a2.jpg", "identity": "person_a", "split": "validation", "session": "day2"},
  {"path": "validation/b2.jpg", "identity": "person_b", "split": "validation", "session": "day2"},
  {"path": "validation/c1.jpg", "identity": "person_c", "split": "validation", "session": "day2"},
  {"path": "test/a3.jpg", "identity": "person_a", "split": "test", "session": "day3"},
  {"path": "test/b3.jpg", "identity": "person_b", "split": "test", "session": "day3"},
  {"path": "test/d1.jpg", "identity": "person_d", "split": "test", "session": "day3"}
]
```

This is a format example, not a supplied dataset or adequate evidence for broad
accuracy claims. Add your real samples. Unknown means the identity is absent from
enrollment; keep its actual pseudonymous label rather than naming everybody Unknown.
The evaluator rejects identical decoded images (including renamed copies), session
overlap for a person, missing known identities and shared validation/test unknowns.
It cannot detect every near-duplicate or verify that session labels are truthful.

## Run

From an activated project environment, after `python -m pip install -e ".[dev]"`:

```sh
evaluate-heldout --manifest data/evaluation/manifest.json --output results/heldout-640.json
```

Enrollment requires exactly one detected face in every image. Validation/test
images with zero or multiple detections remain in the denominator as acquisition
failures; they are not silently dropped or counted as correctly rejected unknowns.
Unreadable files and inference exceptions stop the run rather than hide a broken
pipeline. Each usable image is expected to depict exactly one intended person.

Enrollment embeddings are averaged into normalized centroids using the production
`FaceMatcher`. The threshold is selected **only from validation scores**, maximizing
the mean of known correct identification and unknown correct rejection rates.
Ties favor fewer unknown false accepts and then the higher threshold. The selected
threshold is frozen before computing test metrics. A complete validation sweep and
test metrics at the default threshold are included for comparison.

## Interpret the report

- `known_correct`: accepted with the right identity.
- `known_false_rejects`: an enrolled person rejected below threshold.
- `known_wrong_identity`: accepted as the wrong enrolled person.
- `unknown_false_accepts`: an unenrolled person assigned an enrolled identity.
- `unknown_correct_rejects`: usable unknown face rejected below threshold.
- `*_acquisition_failures`: zero or multiple faces detected.
- `balanced_success_rate`: average of known-correct/all-known and
  unknown-correctly-rejected/all-unknown. Includes acquisition failures as failures.
- `*_rate_given_acquisition`: rates among usable, single-face probes only.
  These are null when no usable probes exist, not a misleading zero.

Always report counts alongside rates, number of identities, capture conditions,
model checksums, threshold and the exact split. Zero observed false accepts on a
small sample is not evidence of zero real-world risk. Images of the same person
are correlated, so this report deliberately does not attach naive independent-trial
confidence intervals. It does not measure demographic fairness or prove suitability
for security-sensitive authentication.

## Compare resolution without tuning on test

Run the same fixed manifest with original-resolution detection:

```sh
evaluate-heldout --manifest data/evaluation/manifest.json --max-input-dimension 0 --output results/heldout-full.json
```

Compare acquisition failures and held-out known/unknown results alongside the
separate timing benchmark. Each mode calibrates on validation only. If choosing
between resolution settings, make that choice from validation and reserve test for
final reporting; repeatedly optimizing against test results invalidates the holdout.

A defensible résumé statement names the actual sample size and conditions. Do not
substitute synthetic unit-test outcomes or enrollment self-matches for measured
recognition performance. The test suite validates logic, not model accuracy.
