from __future__ import annotations

from tas_copilot_core.instrument_aliases import canonical_facility, canonical_instrument, instruments_in, load_registry


def test_registry_is_consistent() -> None:
    assert load_registry().validate() == []


def test_surface_forms_resolve_to_one_instrument() -> None:
    for text in ["BT-7", "BT7", "the BT-7 thermal triple-axis spectrometer", "BT-7 spectrometer at the NCNR", "BT 7 double-focusing thermal triple-axis spectrometer"]:
        m = canonical_instrument(text)
        assert m is not None and m.id == "bt7" and m.facility == "ncnr", text
    assert canonical_instrument("WAND 2 at the HB-2C beamline").id == "hb2c"
    assert canonical_instrument("Quantum Design Physical Property Measurement System (PPMS)").id == "ppms"
    assert canonical_instrument("thermal triple-axis spectrometer Taipan").id == "taipan"
    assert canonical_instrument("a Bruker APEX-II diffractometer") is None  # not in the registry: no guess


def test_lists_every_instrument_in_a_value_in_order() -> None:
    hits = instruments_in("BT1, NOMAD, HB-2A, BT-7, M20 beamline, Quantum Design Physical Property Measurement System (PPMS)")
    assert [h.id for h in hits] == ["bt1", "nomad", "hb2a", "bt7", "ppms"]


def test_facilities_resolve() -> None:
    assert canonical_facility("NIST Center for Neutron Research").id == "ncnr"
    assert canonical_facility("the High Flux Isotope Reactor at ORNL").id == "hfir"
    assert canonical_facility("High Flux Is/Reactor") is None
