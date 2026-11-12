from __future__ import annotations

import numpy as np
import pytest

from tas_copilot_core import counting


def test_counting_uncertainties_come_from_tasreduction_measurement(monkeypatch) -> None:
    pytest.importorskip("tasreduction.uncertainty")
    m = counting.counting_measurement([100.0, 0.0])
    assert type(m).__name__ == "Measurement" and type(m).__module__ == "tasreduction.uncertainty"
    assert np.allclose(m.errors, [10.0, 1.0])  # one sigma, floor of one count on an empty bin

    # counts / time, rescaled: the error scales with the value (Measurement.scale_inplace)
    values, errors = counting.rate_with_uncertainty([400.0, 0.0, 25.0], [2.0, 2.0, 0.0], target=1.0)
    assert np.allclose(values[:2], [200.0, 0.0]) and np.allclose(errors[:2], [10.0, 0.5])
    assert np.isnan(values[2]) and np.isnan(errors[2])  # a zero denominator is not a rate

    # a shift is exact, a scale propagates
    assert np.allclose(counting.scaled_errors([4.0, 6.0], 0.5), [2.0, 3.0])


def test_counting_is_unavailable_without_tasreduction(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("tasreduction"):
            raise ImportError("no tasreduction")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert counting.tasreduction_available() is False
    with pytest.raises(counting.CountingUnavailable, match="tasreduction is not installed"):
        counting.counting_measurement([1.0])
