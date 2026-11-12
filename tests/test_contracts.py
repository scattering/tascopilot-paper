import pytest

from tas_copilot_contracts import (
    AdaptiveCandidatePoint,
    AdaptiveCountRequest,
    AdaptiveCountResult,
    AdaptiveObservation,
    AdaptivePolicyProposal,
    AdaptivePolicyRequest,
    Availability,
    DataFile,
    DerivedDataFile,
    EmbeddingRecord,
    Experiment,
    ImplementationType,
    InstrumentDeviceGraph,
    InstrumentDeviceGraphEdge,
    InstrumentDeviceGraphNode,
    InstrumentGraphAlias,
    Paper,
    PaperExperimentLink,
    PermissionLevel,
    QPoint,
    ReductionCorrectionRef,
    ReductionDAGNode,
    ReductionDAGTemplate,
    ReductionInputRef,
    RepresentationRecord,
    Scan,
    SideEffectLevel,
    SpinWaveCalculationInput,
    SpinWaveCalculationResult,
    SpinWaveExchangeSpec,
    SpinWaveModelSpec,
    SpinWaveSiteSpec,
    TASDataFileMetadata,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    ToolSpec,
    UnitValue,
)


def test_contracts_import_and_validate_core_models():
    assert UnitValue(value=1.0, unit="meV").unit == "meV"
    Paper(paper_id="p1", title="Paper")
    Experiment(experiment_id="e1")
    Scan(scan_id="s1", q_center=QPoint(h=1, k=0, l=0))
    DataFile(datafile_id="d1", path="data/raw/x.bt7")
    TASDataFileMetadata(
        path="data/raw/x.bt7",
        format_family="ncnr_bt7_ice_nice",
        facility="NCNR",
        instrument="BT-7",
        native_format="BT-7 ICE/NICE",
        source_adapter="fixture",
        confidence=0.95,
    )


def test_unit_value_rejects_blank_unit():
    with pytest.raises(ValueError):
        UnitValue(value=1.0, unit=" ")


def test_tool_spec_call_and_result_validation():
    spec = ToolSpec(
        name="read",
        namespace="tas.test",
        version="0.1",
        description="test",
        input_schema={},
        output_schema={},
        side_effect_level=SideEffectLevel.none,
        availability=Availability.available,
        implementation_type=ImplementationType.stub,
        provenance_policy="record everything",
        permissions_required=[PermissionLevel.read_only],
    )
    assert spec.enabled_by_default
    ToolCall(tool_name="read", tool_namespace="tas.test")
    ToolResult(status=ToolResultStatus.success, output={}, tool_name="read", tool_namespace="tas.test")
    ToolResult(status=ToolResultStatus.missing_input, errors=[], tool_name="read", tool_namespace="tas.test")


def test_live_control_tool_cannot_default_enable():
    with pytest.raises(ValueError):
        ToolSpec(
            name="move",
            namespace="tas.bad",
            version="0.1",
            description="bad",
            input_schema={},
            output_schema={},
            side_effect_level=SideEffectLevel.live_control,
            availability=Availability.available,
            implementation_type=ImplementationType.stub,
            provenance_policy="n/a",
            permissions_required=[PermissionLevel.live_control],
            enabled_by_default=True,
        )


def test_mapping_and_representation_contracts():
    PaperExperimentLink(paper_id="p", experiment_id="e", confidence=0.5, evidence=["shared sample"], mapping_method="manual", status="candidate")
    EmbeddingRecord(embedding_id="emb1", source_object_type="scan", source_object_id="s", modality="text", encoder_name="none", encoder_version="0", vector_store_ref="local", vector_dimension=3)
    RepresentationRecord(representation_id="rep1", source_object_type="scan", source_object_id="s", modality="scan_geometry")


def test_instrument_device_graph_contract_preserves_native_aliases():
    node = InstrumentDeviceGraphNode(
        node_id="native:nice_bt7:sampleTwoTheta.softPosition",
        node_kind="motor",
        canonical_role="sample_two_theta",
        native_name="sampleTwoTheta.softPosition",
        native_namespace="nice_bt7",
        aliases=[InstrumentGraphAlias(alias="A4", source_namespace="ice_bt7_column", confidence=0.95)],
    )
    edge = InstrumentDeviceGraphEdge(
        edge_id="e1",
        source_node_id=node.node_id,
        target_node_id="canonical-role:sample_two_theta",
        edge_type="ALIGNED_TO_CANONICAL_ROLE",
        confidence=0.95,
    )
    graph = InstrumentDeviceGraph(
        graph_id="g1",
        source_namespace="nice_bt7",
        nodes=[node],
        edges=[edge],
        canonical_roles=["sample_two_theta"],
    )

    assert graph.nodes[0].native_name == "sampleTwoTheta.softPosition"
    assert graph.nodes[0].aliases[0].alias == "A4"


def test_reduction_dag_contracts_keep_raw_and_correction_inputs_separate():
    raw = ReductionInputRef(input_id="scan-series", source_object_type="scan_group", source_object_id="group-1")
    correction = ReductionCorrectionRef(
        correction_id="monitor",
        correction_type="monitor_normalization",
        status="placeholder",
        stale_ok_for_preview=True,
        parameters={"source": "fixture"},
    )
    template = ReductionDAGTemplate(
        template_id="fixture.template",
        label="fixture",
        measurement_family="elastic_temperature_series",
        nodes=[ReductionDAGNode(node_id="raw", kind="input", operation="input.scan_series")],
    )

    assert raw.immutable is True
    assert correction.status == "placeholder"
    assert "raw inputs are immutable" in template.raw_input_policy


def test_spinwave_generic_model_contract_remains_backward_compatible():
    spec = SpinWaveModelSpec(material="fixture", model_parameters={"j": -1})

    assert spec.kind == "generic"
    assert spec.material == "fixture"
    assert spec.model_parameters["j"] == -1


def test_spinwave_ferromagnetic_chain_contract_matches_pyspinw_adapter_input():
    model = SpinWaveModelSpec(
        kind="ferromagnetic_chain",
        unit_cell=[1, 1, 1],
        site=SpinWaveSiteSpec(position=[0, 0, 0], spin=[0, 0, 1], name="X"),
        exchange=SpinWaveExchangeSpec(type="heisenberg", cell_offset=[1, 0, 0], j=-1),
    )
    request = SpinWaveCalculationInput(
        model=model,
        q_path=[[0, 0, 0], [1, 0, 0]],
        n_points_per_segment=5,
        use_rust=False,
    )
    result = SpinWaveCalculationResult(
        summary="pySpinW ferromagnetic_chain calculation completed.",
        simulation_only=True,
        model_kind="ferromagnetic_chain",
        q_point_count=5,
        energy_shape=[5, 1],
        intensity_shape=[5],
    )

    assert request.model.kind == "ferromagnetic_chain"
    assert request.model.exchange.j == -1
    assert result.simulation_only is True
    assert result.model_kind == "ferromagnetic_chain"


def test_spinwave_ferromagnetic_chain_requires_typed_fields():
    with pytest.raises(ValueError):
        SpinWaveModelSpec(kind="ferromagnetic_chain")

    with pytest.raises(ValueError):
        SpinWaveCalculationInput(
            model=SpinWaveModelSpec(
                kind="ferromagnetic_chain",
                unit_cell=[1, 1, 1],
                site=SpinWaveSiteSpec(),
                exchange=SpinWaveExchangeSpec(),
            ),
            q_path=[[0, 0, 0], [1, 0]],
        )


def test_adaptive_policy_and_count_contracts_match_external_adapter_boundary():
    candidate = AdaptiveCandidatePoint(
        point_id="q-001",
        nodes={"Q.H": 0.1, "Q.K": 0.0, "Q.L": 0.0, "et.deltaE": 5.0},
        metadata={"posterior_variance": 4.0},
    )
    observation = AdaptiveObservation(
        step=0,
        point_id="q-000",
        nodes={"Q.H": 0.0},
        count_time=3.0,
        expected_counts=120.0,
    )
    request = AdaptivePolicyRequest(
        step=1,
        candidates=[candidate],
        observations=[observation],
        model_spec={"default_count_time": 12.0},
        software_limits={"Q.H": {"min": -1.0, "max": 1.0}},
        policy_context={"backend": "external_tasai_adapter"},
    )
    proposal = AdaptivePolicyProposal(
        point_id="q-001",
        nodes=candidate.nodes,
        score=4.2,
        reason="tasai_uncertainty_information_rate",
        count_time=12.0,
    )
    count_request = AdaptiveCountRequest(point=candidate, model_spec=request.model_spec, count_time=2.0)
    count_result = AdaptiveCountResult(expected_counts=40.0, expected_rate_hz=20.0, metadata={"backend": "fixture"})

    assert request.candidates[0].metadata["posterior_variance"] == 4.0
    assert proposal.count_time == 12.0
    assert count_request.point.point_id == "q-001"
    assert count_result.expected_counts == 40.0


def test_derived_data_requires_lineage():
    with pytest.raises(ValueError):
        DerivedDataFile(datafile_id="d2", path="data/processed/x.dat")
