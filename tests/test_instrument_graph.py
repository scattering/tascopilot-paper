import json

from tas_copilot_core.instrument_graph import build_instrument_graph_from_nice_report


def test_nice_report_builds_canonical_graph_without_renaming_native_nodes(tmp_path):
    report = {
        "instrument_dag": {
            "all_nodes": [
                {"id": "Q.H", "deviceID": "Q", "description": "Miller index H", "userUnit": "", "dataType": "java.lang.Double", "hasControlInput": True, "storageMode": "STATE"},
                {"id": "Q.K", "deviceID": "Q", "description": "Miller index K", "userUnit": "", "dataType": "java.lang.Double", "hasControlInput": True, "storageMode": "STATE"},
                {"id": "Q.L", "deviceID": "Q", "description": "Miller index L", "userUnit": "", "dataType": "java.lang.Double", "hasControlInput": True, "storageMode": "STATE"},
                {"id": "Q.length", "deviceID": "Q", "description": "Q magnitude", "userUnit": "1/Å", "dataType": "java.lang.Double", "hasControlInput": True, "storageMode": "STATE"},
                {"id": "Q.powderMode", "deviceID": "Q", "description": "Moving H,K,L will only move a4 when set to true", "userUnit": "", "dataType": "java.lang.Boolean", "hasControlInput": True, "storageMode": "STATIC"},
                {"id": "et.deltaE", "deviceID": "et", "description": "Energy transfer", "userUnit": "meV", "dataType": "java.lang.Double", "hasControlInput": True, "storageMode": "STATE"},
                {"id": "et.fixedEnergyMode", "deviceID": "et", "description": "Fixed energy mode", "userUnit": "", "dataType": "java.lang.String", "hasControlInput": True, "storageMode": "STATE"},
                {"id": "ei.energy", "deviceID": "ei", "description": "Incident energy", "userUnit": "meV", "dataType": "java.lang.Double", "hasControlInput": True, "storageMode": "STATE"},
                {"id": "ef.energy", "deviceID": "ef", "description": "Final energy", "userUnit": "meV", "dataType": "java.lang.Double", "hasControlInput": True, "storageMode": "STATE"},
                {"id": "sampleTheta.softPosition", "deviceID": "sampleTheta", "description": "A3", "userUnit": "°", "dataType": "java.lang.Double", "hasControlInput": True, "storageMode": "STATE"},
                {"id": "sampleTwoTheta.softPosition", "deviceID": "sampleTwoTheta", "description": "A4", "userUnit": "°", "dataType": "java.lang.Double", "hasControlInput": True, "storageMode": "STATE"},
                {"id": "counter.liveROI", "deviceID": "counter", "description": "ROI", "userUnit": "", "dataType": "java.lang.Integer", "hasControlInput": False, "storageMode": "COUNTS"},
                {"id": "counter.liveMonitor", "deviceID": "counter", "description": "Monitor", "userUnit": "", "dataType": "java.lang.Integer", "hasControlInput": False, "storageMode": "COUNTS"},
                {"id": "singleDetector.counts", "deviceID": "singleDetector", "description": "Detector counts", "userUnit": "", "dataType": "java.util.List<java.lang.Integer>", "hasControlInput": False, "storageMode": "COUNTS"},
            ]
        }
    }
    report_path = tmp_path / "nice_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    graph = build_instrument_graph_from_nice_report(report_path)

    roles = set(graph.canonical_roles)
    assert {"q_h", "q_k", "q_l", "energy_transfer", "sample_two_theta", "roi_readback"} <= roles
    native = {node.native_name: node for node in graph.nodes if node.native_namespace == "nice_bt7"}
    assert native["sampleTwoTheta.softPosition"].canonical_role == "sample_two_theta"
    assert native["sampleTwoTheta.softPosition"].native_name == "sampleTwoTheta.softPosition"
    assert any(edge.edge_type == "ALIGNED_TO_CANONICAL_ROLE" for edge in graph.edges)
    assert any(edge.edge_type == "DEPENDS_ON" for edge in graph.edges)
    assert any("Native names are preserved" in warning for warning in graph.warnings)

