"""Regression tests. Run with: pytest -q tests"""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "code"))

from generate_sequence import SequenceSpec, build, match_composition
from structure_factor_check import structure_factor, fit_alpha


def _alpha(source, n=60000, seed=0):
    d = build(SequenceSpec(source, n, seed=seed))
    k0 = 2 * np.pi / 1.25
    k, s = structure_factor(d["eps"], d["x_nm"], taper="hann", k0=k0)
    return fit_alpha(k, s / d["g2"], k0)[0]


def test_class_ii_alpha_is_one():
    """GUE surrogate must recover alpha = 1 (Eq. 9, class II)."""
    assert abs(_alpha("gue") - 1.0) < 0.15


def test_random_alloy_alpha_is_zero():
    """Uncorrelated alloy is the alpha = 0 control."""
    assert abs(_alpha("random")) < 0.15


def test_g2_matches_analytic():
    d = build(SequenceSpec("random", 40000))
    c = d["concentration"]
    m10, m11 = 10.012937, 11.009305
    mbar = c * m10 + (1 - c) * m11
    g2 = c * (1 - m10 / mbar) ** 2 + (1 - c) * (1 - m11 / mbar) ** 2
    assert abs(d["g2"] - g2) / g2 < 1e-6


def test_composition_matching_is_exact():
    rng = np.random.default_rng(1)
    occ = rng.random(10000) < 0.3
    occ = match_composition(occ, 0.2, rng, warn_frac=1.0)
    assert occ.sum() == 2000


def test_discretisation_background_formula():
    """Eq. (21) must agree with the exact 1 - sinc^2 to 6% for a/ell = 0.2."""
    a, ell = 0.2504, 1.25
    k0 = 2 * np.pi / ell
    exact = 1 - np.sinc(k0 * a / (2 * np.pi)) ** 2
    expansion = np.pi ** 2 / 3 * (a / ell) ** 2
    assert abs(expansion - exact) / exact < 0.06


def test_riemann_siegel_zeros():
    """Zero finder must reproduce the first five ordinates to <1e-2 absolute,
    which is >100x below the lattice rounding quantum in unfolded units."""
    import riemann_siegel as rs
    known = np.array([14.134725142, 21.022039639, 25.010857580,
                      30.424876126, 32.935061588])
    assert np.abs(rs.zeros(5) - known).max() < 1e-2


def test_zeta_spacing_rigidity_increases_with_height():
    """
    Low-lying zeros are more rigid than GUE; sigma_s -> 0.4180 from below.

    This is the *supported* half of the finite-height result.  There is
    deliberately no test asserting a height dependence of alpha: that trend is
    only 2.1 sigma in the six-point run and a constant alpha fits the data
    (chi2/dof = 0.81).  See the README.
    """
    import riemann_siegel as rs
    lo = np.diff(rs.unfold(rs.zeros(2000, start_index=0))).std()
    hi = np.diff(rs.unfold(rs.zeros(2000, start_index=100000))).std()
    assert lo < hi < 0.4180


def test_overlap_guard_rejects_the_old_ladder():
    """The first 24-window run used overlapping heights; that must now fail."""
    from fig3_alpha_vs_height import check_overlap, auto_heights
    old = [0, 5000, 20000, 60000, 150000, 400000, 1000000, 2500000]
    try:
        check_overlap(old, 12000, 24, strict=True)
        raise AssertionError("overlapping ladder was not rejected")
    except SystemExit:
        pass
    check_overlap(auto_heights(8, 12000, 24), 12000, 24, strict=True)


def test_trend_statistics_use_unscaled_covariance():
    """
    Slope significance from the WLS covariance must equal the likelihood-ratio
    Delta chi2. numpy.polyfit(cov=True) rescales and would not agree.
    """
    from fig3_alpha_vs_height import weighted_trend
    a = np.array([2.022, 2.011, 1.761, 1.545, 1.435, 1.261, 1.378, 1.420])
    e = np.array([0.272, 0.260, 0.295, 0.333, 0.344, 0.376, 0.390, 0.366])
    g = np.sqrt(np.array([14, 5449, 18047, 47532, 107719, 260877, 600270, 1389294.])
                * np.array([93737, 96999, 106726, 132237, 187999, 335852,
                            670314, 1454755.]))
    r = weighted_trend(a, e, g)
    assert abs(r["sigma_slope"] - r["sigma_dchi2"]) < 0.15
    assert r["sigma_slope"] < 3.0          # the trend is NOT established
    assert r["sigma_above_one"] > 3.0      # but alpha > 1 is
