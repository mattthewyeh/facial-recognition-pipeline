import numpy as np
import pytest

from face_pipeline.benchmark import measure_frame, summarize_runs
from face_pipeline.measurement import summarize_ms


class Detector:
    def __init__(self, count=1): self.count = count
    def detect(self, frame): return [object()] * self.count


class Embedder:
    def align(self, frame, detection): return frame
    def extract_from_aligned(self, frame): return np.array([1., 0.])


class Matcher:
    def match(self, vector): return None


def clock():
    from itertools import count
    ticks = count()
    return lambda: next(ticks) / 1000


def test_stage_timings_and_total_include_actual_matching():
    result = measure_frame(None, Detector(), Embedder(), Matcher(), clock=clock())
    assert result['status'] == 'ok'
    assert set(result['timings']) == {'detection', 'alignment', 'embedding', 'matching', 'total'}
    assert result['timings']['detection'] == pytest.approx(1)
    assert result['timings']['total'] >= sum(v for k,v in result['timings'].items() if k != 'total')


def test_aligned_mode_does_not_claim_detection_or_alignment_time():
    result = measure_frame(None, None, Embedder(), aligned=True, clock=clock())
    assert set(result['timings']) == {'embedding', 'total'}


def test_failed_detections_do_not_make_successful_pipeline_look_faster():
    good = measure_frame(None, Detector(), Embedder(), clock=clock())
    bad = measure_frame(None, Detector(0), Embedder(), clock=clock())
    multi = measure_frame(None, Detector(2), Embedder(), clock=clock())
    summary = summarize_runs([good, bad, multi])
    assert summary['outcomes'] == {'ok': 1, 'no_face': 1, 'multiple_faces': 1}
    assert summary['all_attempts_total']['count'] == 3
    assert summary['successful_attempts']['total']['count'] == 1
    assert summary['successful_attempts']['matching']['count'] == 0


def test_percentiles_and_empty_measurements():
    assert summarize_ms([1, 2, 3, 4, 100])['median_ms'] == 3
    assert summarize_ms([1, 2, 3, 4, 100])['p95_ms'] == pytest.approx(80.8)
    assert summarize_ms([])['median_ms'] is None
    with pytest.raises(ValueError): summarize_ms([float('nan')])


def test_benchmark_cli_excludes_warmup_and_alternates_configs(tmp_path, monkeypatch):
    import sys
    import cv2 as cv
    import json
    from face_pipeline import benchmark
    image_dir = tmp_path / 'images'
    image_dir.mkdir()
    cv.imwrite(str(image_dir / 'frame.png'), np.zeros((20, 30, 3), dtype=np.uint8))
    order = []
    monkeypatch.setattr(benchmark, 'YuNetFaceDetector', lambda path, max_input_dimension: max_input_dimension)
    monkeypatch.setattr(benchmark, 'SFaceEmbedder', lambda path: None)
    monkeypatch.setattr(benchmark, 'environment', lambda *args: {})
    monkeypatch.setattr(cv, 'getNumThreads', lambda: 8)
    def measure(frame, detector, *args, **kwargs):
        order.append(detector)
        return {'status': 'ok', 'timings': {'total': float(detector)}}
    monkeypatch.setattr(benchmark, 'measure_frame', measure)
    out = tmp_path / 'out.json'
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--images', str(image_dir), '--output', str(out),
                                    '--warmup', '1', '--repeats', '2', '--dimensions', '0', '10'])
    benchmark.main()
    report = json.loads(out.read_text())
    assert order == [30, 10, 10, 30, 30, 10]
    assert report['configurations']['0']['outcomes']['ok'] == 2
    assert report['comparisons']['10']['paired_successes'] == 2
    assert report['comparisons']['10']['median_speedup'] == 3
    assert report['comparisons']['10']['images_actually_resized'] == 1
    assert str(tmp_path) not in out.read_text()
    assert 'reports 8' in report['warnings'][0]
