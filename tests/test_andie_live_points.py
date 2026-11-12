"""Live points through NICE's own scripted measurement (2026-10-31)."""
import pytest

from test_andie_nice_integration import (NativeFileAPI, bridge_session, full_adapter,
                                         install_inprocess_server, load_mocked_server,
                                         native_request)


class MeasurementAPI(NativeFileAPI):
    """NICE's measurement_start/count/end; each count updates the counters."""

    def __init__(self, data_root):
        super().__init__(data_root)
        self.measurements = []
        self.counts = []

    def measurement_start(self, control_nodes, title="", xAxis="", yAxis="", norm="",
                          customVars=None, num_points=1):
        self.measurements.append({"start": list(control_nodes), "title": title, "xAxis": xAxis,
                                  "yAxis": yAxis, "num_points": num_points})

    def measurement_count(self, filePrefix=None, presetMonitor=-1, presetTime=-1, **kwargs):
        i = len(self.counts)
        self.counts.append({"filePrefix": filePrefix, "presetMonitor": presetMonitor,
                            "presetTime": presetTime})
        self.nodes.update({"counter.liveROI": 60 + i, "counter.liveMonitor": 39048,
                           "counter.liveTime": 30.0 + 0.01 * i, "singleDetector": [10, 40 + i, 10],
                           "temp": 105.0 + 0.001 * i})

    def measurement_end(self):
        self.measurements.append("end")


def live_request():
    req = native_request()
    req.instrument_config["nice_acquisition"] = "measurement"
    req.instrument_config["nice_live"] = {"counts": "counter.liveROI", "monitor": "counter.liveMonitor",
                                          "live_time": "counter.liveTime", "segments": "singleDetector",
                                          "temperature": "temp"}
    return req


def test_live_points_use_nice_measurement_and_read_each_point(monkeypatch, tmp_path):
    api = MeasurementAPI(tmp_path / "data")
    server = load_mocked_server(monkeypatch, api)
    processes = install_inprocess_server(monkeypatch, server, api)
    adapter = full_adapter(bridge_session(tmp_path / "spool"), api, tmp_path / "actions")
    req = live_request()
    try:
        adapter.start()
        result = adapter.execute(req)
        assert result.status == "complete", result.reason
        assert api.trajectories == []                      # no trajectory, no file to wait for
        starts = [m for m in api.measurements if m != "end"]
        assert len(starts) == 2 and api.measurements.count("end") == 2
        assert set(starts[0]["start"]) == {"sampleTwoTheta.softPosition", "sampleTheta.softPosition"}
        assert starts[0]["xAxis"] == "sampleTwoTheta.softPosition"
        assert starts[0]["yAxis"] == "singleDetector.liveROI"
        assert len(api.counts) == 62                       # 31 points x 2 scans
        assert all(c["presetMonitor"] == 39048 for c in api.counts)
        index = 0
        for spec, scan in zip(req.scans, result.scans):
            assert len(scan.rows) == 31
            prefixes = {c["filePrefix"] for c in api.counts[index:index + 31]}
            assert len(prefixes) == 1 and next(iter(prefixes)).startswith("SyntheticSample_")
            for k, row in enumerate(scan.rows):
                assert row.requested == spec.points[k]
                assert row.achieved == pytest.approx(spec.points[k])   # read back after the move
                assert row.counts == 60 + index + k
                assert row.detector_segments == [10, 40 + index + k, 10]
                assert row.monitor == 39048
                assert row.temperature_k == pytest.approx(105.0 + 0.001 * (index + k))
            index += 31
        progress = sorted(adapter.session.base_spool.glob("**/progress/*.jsonl"))
        assert len(progress) == 2
        lines = progress[0].read_text().splitlines()
        assert len(lines) == 31 and '"of": 31' in lines[0]
    finally:
        adapter.stop()
    assert processes[0].failure is None
