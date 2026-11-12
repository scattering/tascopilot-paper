from __future__ import annotations

import pytest

from tas_copilot_core.magnetic_form_factor import apply_form_factor, magnetic_form_factor

pytest.importorskip("periodictable")


def test_form_factor_decreases_monotonically_over_the_measurable_range() -> None:
    """A free-ion f(Q) is the FT of a smooth positive spin density: it only decays.

    This is the property that was violated in practice, because pySpinW applies
    no form factor at all and the legacy placeholder was a hard-coded
    exp(-0.1 Q^2) with no ion dependence.
    """
    q = [0.25 * n for n in range(1, 33)]  # 0.25 .. 8 1/A, the thermal-TAS range
    report = magnetic_form_factor("Mn2+", q)
    f = report["form_factor"]

    assert f[0] > 0.99, "f(Q) -> 1 as Q -> 0"
    assert all(later < earlier for earlier, later in zip(f, f[1:])), "must decrease monotonically"
    assert report["monotonic_over_requested_q"] is True
    assert report["warnings"] == []
    # Intensity carries f^2, which must fall faster than f.
    assert report["form_factor_squared"][10] < f[10]


def test_coefficients_are_the_standard_mn2_plus_values() -> None:
    """Guards against a hand-copied or invented table.

    The placeholder we replaced used exp(-0.1 Q^2) for every ion; these are the
    real tabulated <j0> coefficients, and c is negative, which is what makes the
    parameterisation turn up far outside its fitted range.
    """
    report = magnetic_form_factor("Mn2+", [1.0])
    a, aa, b, bb, c, cc, d = report["j0_coefficients"]
    assert (round(a, 4), round(aa, 3)) == (0.422, 17.684)
    assert (round(b, 4), round(bb, 3)) == (0.5948, 6.005)
    assert cc < 0, "the C-term exponent is negative; it grows at large Q"
    assert d < 0, "the residual constant is negative, so f(Q) crosses zero eventually"


def test_upturn_beyond_the_fitted_range_is_flagged_not_hidden() -> None:
    """The high-Q upturn is a fitting artifact and must be reported as such.

    It is never observed: |f| < 0.02 there, so f^2 < 4e-4, orders below the
    Q->0 signal and far under phonon background.
    """
    report = magnetic_form_factor("Mn2+", [8.0, 10.0, 12.0])
    assert report["warnings"], "beyond-range Q must warn"
    assert "artifact" in report["warnings"][0]

    # f itself keeps decreasing - it crosses BELOW zero near 9.7 1/A, which a
    # true form factor never does (it approaches zero from above). The visible
    # upturn is therefore in f^2, the quantity that multiplies intensity:
    # it bottoms out around Q=10 and climbs again.
    f = report["form_factor"]
    assert f[1] < 0 and f[2] < 0, "the residual constant D drags f below zero"
    f2 = report["form_factor_squared"]
    assert f2[1] < f2[0] and f2[2] > f2[1], "f^2 has a minimum then turns up"

    # And it is unobservable: four orders below the Q->0 intensity, far under
    # phonon background, which is why no measurement shows this.
    assert max(f2[1], f2[2]) < 1.0e-3


def test_j2_orbital_term_is_optional_and_off_by_default() -> None:
    """<j2> is intrinsically non-monotonic and must not be added blindly.

    It is zero at Q=0, peaks near 5 1/A, then decays - real physics for ions
    with orbital moment, but Mn2+ is 3d5 (L=0) and spin-only.
    """
    spin_only = magnetic_form_factor("Mn2+", [3.0])
    assert spin_only["includes_j2_orbital_term"] is False
    with_orbital = magnetic_form_factor("Mn2+", [3.0], include_j2=True, g_factor=2.0)
    # At g=2 the dipole prefactor (1 - 2/g) vanishes, so the two agree.
    assert with_orbital["form_factor"][0] == pytest.approx(spin_only["form_factor"][0])
    # With g != 2 the orbital term contributes.
    tilted = magnetic_form_factor("Mn2+", [3.0], include_j2=True, g_factor=1.5)
    assert tilted["form_factor"][0] != pytest.approx(spin_only["form_factor"][0])


def test_apply_form_factor_multiplies_intensity_by_f_squared() -> None:
    """Intensity carries f^2, not f - the legacy code multiplied Sab by f once."""
    result = apply_form_factor([100.0, 100.0, 100.0], [1.0, 3.0, 6.0], ion="Mn2+")
    corrected = result["corrected_intensities"]
    assert corrected[0] > corrected[1] > corrected[2]
    assert corrected[0] == pytest.approx(100.0 * magnetic_form_factor("Mn2+", [1.0])["form_factor"][0] ** 2)

    with pytest.raises(ValueError, match="same length"):
        apply_form_factor([1.0, 2.0], [1.0], ion="Mn2+")


def test_ion_parsing_and_unknown_ions() -> None:
    assert magnetic_form_factor("Mn2+", [1.0])["form_factor"] == magnetic_form_factor("Mn+2", [1.0])["form_factor"]
    with pytest.raises(ValueError):
        magnetic_form_factor("Zz9+", [1.0])
    with pytest.raises(ValueError):
        magnetic_form_factor("Mn2+", [-1.0])
