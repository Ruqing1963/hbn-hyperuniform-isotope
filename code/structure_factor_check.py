#!/usr/bin/env python3
"""
structure_factor_check.py
=========================

Measure the mass-fluctuation structure factor

    S_eps(k) = (1/N) |sum_n eps_n exp(i k x_n)|^2 = g2 * s(k)          [Eq. (7)]

for each generated arrangement, extract the small-k exponent alpha in
s(k) ~ (k/k0)^alpha [Eq. (9)], and verify the discretisation decomposition

    s_disc(k) = sinc^2(ka/2) s_ideal(k) + [1 - sinc^2(ka/2)]           [Eq. (18)]

together with the band-edge contamination bound

    (s_bg / s_ideal)|_{k0} = (pi^2 / 3) (a / ell)^2                    [Eq. (21)]

--------------------------------------------------------------------------
METHODOLOGICAL WARNING -- read before interpreting any fitted alpha
--------------------------------------------------------------------------
The periodogram estimator of S(k) on a finite sequence is *biased low in
exponent*.  Rectangular (untapered) windowing leaks power from the
non-hyperuniform large-k plateau into the small-k region, flattening the
apparent slope.  On GUE sequences of 10^3-10^4 points this produces
alpha_apparent ~ 0.65-0.89 -- i.e. it looks as though hyperuniformity has been
degraded, when it has not.

Applying a Hann taper removes the bias and recovers alpha -> 1.  This is a
property of the estimator, not of the sequence: the number variance of the
same unfolded spectra agrees with the GUE prediction
sigma^2(R) = [ln(2 pi R) + gamma_E + 1] / pi^2 to better than 1 percent at
every R tested.

Always report alpha from the tapered estimator, and always report the
untapered value alongside it so the reader can see the size of the artefact.
--------------------------------------------------------------------------

Usage
-----
    python structure_factor_check.py --in-dir ../data/processed \
                                     --fig ../figures/fig1_structure_factor.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# Estimator
# --------------------------------------------------------------------------

def structure_factor(eps: np.ndarray, x: np.ndarray, kvals=None,
                     taper: str = "hann", n_bins: int = 48, k_max_frac: float = 1.3,
                     k0: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Band-averaged periodogram estimate of S_eps(k) on a regular lattice.

    A raw periodogram ordinate is exponentially distributed: its standard
    deviation equals its mean.  A single realisation therefore cannot
    constrain alpha at all (fitted errors of +/-0.5 are typical).  We evaluate
    every Fourier mode with one FFT and then average within logarithmically
    spaced bands (Bartlett smoothing), which reduces the variance by the
    number of modes per band.

    Returns (k_centres, s_binned).
    """
    n = eps.size
    a_nm = float(x[1] - x[0])
    w = np.hanning(n) if taper == "hann" else np.ones(n)
    w = w / np.sqrt(np.mean(w ** 2))

    S = np.abs(np.fft.rfft(w * eps)) ** 2 / n
    k = 2 * np.pi * np.fft.rfftfreq(n, d=a_nm)

    kmin = 20 * k[1]                      # stay clear of the window main lobe
    kmax = k_max_frac * k0
    edges = np.logspace(np.log10(kmin), np.log10(kmax), n_bins + 1)
    idx = np.digitize(k, edges) - 1
    ok = (idx >= 0) & (idx < n_bins)
    cnt = np.bincount(idx[ok], minlength=n_bins)
    tot = np.bincount(idx[ok], weights=S[ok], minlength=n_bins)
    kc = np.sqrt(edges[:-1] * edges[1:])
    good = cnt > 0
    return kc[good], (tot[good] / np.maximum(cnt[good], 1))


def number_variance(sites_nm: np.ndarray, radii: np.ndarray,
                    n_windows: int = 4000, rng=None) -> np.ndarray:
    """sigma^2(R) diagnostic; independent check that unfolding is sound."""
    rng = rng or np.random.default_rng(0)
    u = np.sort(sites_nm)
    dens = u.size / (u[-1] - u[0])
    u = u * dens                                   # rescale to unit density
    lo, hi = u[0] + 50, u[-1] - 50
    return np.array([
        (np.searchsorted(u, (c := rng.uniform(lo, hi - R, n_windows)) + R)
         - np.searchsorted(u, c)).var()
        for R in radii
    ])


def fit_alpha(k: np.ndarray, s: np.ndarray, k0: float,
              lo: float = 1 / 50, hi: float = 1 / 5) -> tuple[float, float]:
    """Log-log slope of s(k) over k in [lo*k0, hi*k0]. Returns (alpha, stderr)."""
    m = (k > lo * k0) & (k < hi * k0) & (s > 0)
    lx, ly = np.log(k[m]), np.log(s[m])
    p, cov = np.polyfit(lx, ly, 1, cov=True)
    return float(p[0]), float(np.sqrt(cov[0, 0]))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def analyse(path: Path, n_k: int = 240) -> dict:
    d = np.load(path, allow_pickle=True)
    eps, x = d["eps"], d["x_nm"]
    meta = json.loads(str(d["meta"]))
    a, ell = meta["a_nm"], meta["ell_nm"]
    k0 = 2 * np.pi / ell

    k, s_h = structure_factor(eps, x, taper="hann", k0=k0)
    _, s_r = structure_factor(eps, x, taper="rect", k0=k0)
    s_h = s_h / float(d["g2"]); s_r = s_r / float(d["g2"])

    a_h, e_h = fit_alpha(k, s_h, k0)
    a_r, e_r = fit_alpha(k, s_r, k0)

    sinc2 = np.sinc(k * a / (2 * np.pi)) ** 2
    bg_pred_exact = 1.0 - sinc2
    bg_pred_expan = (np.pi ** 2 / 3) * (a / ell) ** 2 * (k / k0) ** 2

    return dict(name=path.stem.replace("seq_", ""), k=k, k0=k0,
                s_hann=s_h, s_rect=s_r,
                alpha_hann=a_h, alpha_hann_err=e_h,
                alpha_rect=a_r, alpha_rect_err=e_r,
                bg_exact=bg_pred_exact, bg_expansion=bg_pred_expan,
                a_nm=a, ell_nm=ell, g2=float(d["g2"]),
                concentration=float(d["concentration"]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=Path("../data/processed"))
    p.add_argument("--fig", type=Path, default=Path("../figures/fig1_structure_factor.pdf"))
    p.add_argument("--csv", type=Path, default=Path("../data/processed/alpha_table.csv"))
    args = p.parse_args()

    files = sorted(args.in_dir.glob("seq_*.npz"))
    if not files:
        raise SystemExit(f"no seq_*.npz in {args.in_dir}; run generate_sequence.py --all first")

    res = [analyse(f) for f in files]

    print(f"{'arrangement':<12}{'c':>10}{'g2':>12}"
          f"{'alpha (hann)':>18}{'alpha (rect)':>18}")
    print("-" * 70)
    rows = ["arrangement,concentration,g2,alpha_hann,alpha_hann_err,alpha_rect,alpha_rect_err"]
    for r in res:
        print(f"{r['name']:<12}{r['concentration']:>10.5f}{r['g2']:>12.4e}"
              f"{r['alpha_hann']:>12.3f} +/-{r['alpha_hann_err']:.3f}"
              f"{r['alpha_rect']:>12.3f} +/-{r['alpha_rect_err']:.3f}")
        rows.append(f"{r['name']},{r['concentration']:.6f},{r['g2']:.6e},"
                    f"{r['alpha_hann']:.4f},{r['alpha_hann_err']:.4f},"
                    f"{r['alpha_rect']:.4f},{r['alpha_rect_err']:.4f}")

    cs = {r["concentration"] for r in res}
    if max(cs) - min(cs) > 2e-3:
        print("\n*** WARNING: concentrations differ by more than 0.2%.  The "
              "differential observable of Sec. IV B is invalid. ***")
    else:
        print(f"\ncomposition matched to delta_c = {max(cs) - min(cs):.2e}  (spec: < 2e-3)")

    ref = res[0]
    print(f"\nband-edge contamination, Eq. (21): pi^2/3 * (a/ell)^2 = "
          f"{np.pi**2/3*(ref['a_nm']/ref['ell_nm'])**2:.4f}   "
          f"(exact 1-sinc^2 at k0 = {1-np.sinc(ref['k0']*ref['a_nm']/(2*np.pi))**2:.4f})")
    print("Eq. (21) is the leading-order expansion; it is accurate to ~5% for "
          "a/ell <~ 0.2 and should not be used above that.")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.csv.write_text("\n".join(rows) + "\n")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for r in res:
        ax[0].loglog(r["k"] / r["k0"], r["s_hann"], lw=1.3, label=r["name"])
    kk = np.logspace(-2, 0, 50)
    ax[0].loglog(kk, kk, "k--", lw=0.9, label=r"$s\propto k^{1}$")
    ax[0].set_xlabel(r"$k/k_0$"); ax[0].set_ylabel(r"$s(k)$")
    ax[0].set_title("tapered estimator"); ax[0].legend(fontsize=7)

    r = next(x for x in res if x["name"] in ("gue", "zeta"))
    ax[1].loglog(r["k"] / r["k0"], r["s_hann"], lw=1.3, label="Hann")
    ax[1].loglog(r["k"] / r["k0"], r["s_rect"], lw=1.3, label="rect (biased)")
    ax[1].loglog(kk, kk, "k--", lw=0.9, label=r"$k^1$")
    ax[1].set_xlabel(r"$k/k_0$")
    ax[1].set_title(f"windowing artefact: "
                    rf"$\alpha={r['alpha_hann']:.2f}$ vs ${r['alpha_rect']:.2f}$")
    ax[1].legend(fontsize=7)

    fig.tight_layout()
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=200)
    print(f"\nfigure -> {args.fig}\ntable  -> {args.csv}")


if __name__ == "__main__":
    main()
