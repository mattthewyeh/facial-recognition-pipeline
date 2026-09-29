import json
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from face_pipeline.heldout import Observation, build_report, metrics, select_threshold
from face_pipeline.protocol import load_manifest, dataset_summary


def manifest(tmp_path):
    rows = []
    specs = [('a', 'enrollment'), ('a', 'validation'), ('u', 'validation'), ('a', 'test'), ('v', 'test')]
    for i, (identity, split) in enumerate(specs):
        name = f'{i}.png'
        cv.imwrite(str(tmp_path / name), np.full((12, 12, 3), i * 40, dtype=np.uint8))
        rows.append(dict(path=name, identity=identity, split=split, session=split))
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(rows))
    return path, rows


def test_manifest_resolves_relative_paths_and_aggregates_without_names(tmp_path):
    path, _ = manifest(tmp_path)
    samples = load_manifest(path)
    assert len(samples) == 5
    assert samples[0].path.is_absolute()
    report = dataset_summary(samples)
    assert report['splits']['test'] == {'images': 2, 'identities': 2}
    assert str(tmp_path) not in json.dumps(report)
    assert report == dataset_summary(list(reversed(samples)))


@pytest.mark.parametrize('change, error', [
    ('duplicate', 'duplicate'), ('session', 'session crosses'),
    ('unknown_overlap', 'disjoint'), ('missing_known', 'every enrolled'),
    ('missing_unknown', 'unenrolled'), ('bad_split', 'invalid split'),
    ('unreadable', 'unreadable'), ('missing_field', 'required strings'),
])
def test_manifest_rejects_leakage_and_invalid_protocol(tmp_path, change, error):
    path, rows = manifest(tmp_path)
    if change == 'duplicate': rows[1]['path'] = rows[0]['path']
    if change == 'session': rows[1]['session'] = rows[0]['session']
    if change == 'unknown_overlap': rows[4]['identity'] = 'u'
    if change == 'missing_known': rows[3]['identity'] = 'b'
    if change == 'missing_unknown': rows[4]['identity'] = 'a'
    if change == 'bad_split': rows[4]['split'] = 'training'
    if change == 'unreadable': rows[4]['path'] = 'missing.png'
    if change == 'missing_field': del rows[0]['session']
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match=error): load_manifest(path)


def test_metrics_distinguish_wrong_matches_rejections_and_detection_failures():
    rows = [Observation('a', 'a', .9), Observation('a', 'a', .2), Observation('a', 'b', .8),
            Observation('a', None, None, 'no_face'), Observation('u', 'b', .9),
            Observation('u', 'a', .1), Observation('u', None, None, 'multiple_faces')]
    result = metrics(rows, {'a', 'b'}, .5)
    assert result['known_correct'] == result['known_false_rejects'] == result['known_wrong_identity'] == 1
    assert result['known_acquisition_failures'] == result['unknown_acquisition_failures'] == 1
    assert result['unknown_false_accepts'] == result['unknown_correct_rejects'] == 1
    assert result['balanced_success_rate'] == pytest.approx((1/4 + 1/3)/2)
    assert result['unknown_false_accept_rate_given_acquisition'] == .5
    assert result['known_false_reject_rate_given_acquisition'] == pytest.approx(1/3)


def test_threshold_selection_cannot_use_test_outcomes():
    validation = [Observation('a', 'a', .8), Observation('u', 'a', .2)]
    first = build_report({'validation': validation, 'test': [Observation('a', 'a', .9), Observation('v', 'a', .1)]}, {'a'})
    second = build_report({'validation': validation, 'test': [Observation('a', 'a', .1), Observation('v', 'a', .99)]}, {'a'})
    assert first['selected_threshold'] == second['selected_threshold']
    assert first['test']['balanced_success_rate'] == 1
    assert second['test']['balanced_success_rate'] == 0


def test_threshold_inclusive_boundary_and_failed_acquisition_not_success():
    result = metrics([Observation('a', 'a', .5), Observation('u', None, None, 'no_face')], {'a'}, .5)
    assert result['known_correct'] == 1
    assert result['unknown_correct_rejects'] == 0
    assert result['unknown_false_accept_rate_given_acquisition'] is None


def test_calibration_requires_both_successful_classes():
    with pytest.raises(ValueError, match='successful'):
        select_threshold([Observation('a', 'a', .9), Observation('u', None, None, 'no_face')], {'a'})


@pytest.mark.parametrize('threshold', [float('nan'), float('inf'), -2, 2])
def test_invalid_thresholds_rejected(threshold):
    with pytest.raises(ValueError): metrics([], {'a'}, threshold)


def test_enrollment_centroids_and_probe_failures_use_production_matcher(tmp_path):
    from face_pipeline.heldout import gallery, observe
    path, _ = manifest(tmp_path)
    samples = load_manifest(path)
    class Detector:
        def detect(self, frame):
            return [] if int(frame[0,0,0]) == 160 else [object()]
    class Embedder:
        def extract(self, frame, face):
            return np.asarray([1., 0.]) if int(frame[0,0,0]) < 80 else np.asarray([0., 1.])
    profiles = gallery(samples, Detector(), Embedder())
    observations = observe(samples, profiles, Detector(), Embedder())
    assert len(profiles) == 1
    assert profiles[0].centroid.tolist() == [1, 0]
    assert observations['validation'][0].predicted == 'a'
    assert observations['validation'][0].similarity == 1
    assert observations['test'][1].status == 'no_face'
    assert observations['test'][1].similarity is None


def test_enrollment_failure_aborts_instead_of_weakening_gallery(tmp_path):
    from face_pipeline.heldout import gallery
    path, _ = manifest(tmp_path)
    class Detector:
        def detect(self, frame): return []
    with pytest.raises(ValueError, match='enrollment image'):
        gallery(load_manifest(path), Detector(), None)
