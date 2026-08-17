#!/usr/bin/env python3
"""
fig3_alpha_vs_height.py
=======================

Figure 3: the hyperuniformity exponent alpha of a lattice-mapped Riemann-zero
sequence versus the height of the zeros used, together with the
nearest-neighbour spacing dispersion over the same windows.

Physics
-------
The Montgomery-Odlyzko correspondence -- that the unfolded zeta zeros share
the pair statistics of the GUE -- is asymptotic in height, and the approach is
only logarithmic.  Finite-height deviations are themselves well known in the
random-matrix and quantum-chaos literature (semiclassical prime-pair
corrections to the spectral form factor); Odlyzko's clean numerical agreement
required zeros near 10^20.

What this script measures is the *size* of that deviation at realizable
modulation lengths, and whether it varies enough with height to be usable as a
design parameter.

--------------------------------------------------------------------------
WINDOW OVERLAP -- the defect this version fixes
--------------------------------------------------------------------------
Each height point averages `n_windows` CONSECUTIVE blocks of `n_pts` zeros, so
one point consumes indices [n0, n0 + n_windows*n_pts).  If two starting
indices are closer together than that span, the two points share most of their
zeros: their errors are correlated and any fitted trend is partly an artefact.

In the first 24-window run the starts 0, 5000, 20000, 60000 overlapped by
100/96/84/53 percent, leaving only five genuinely independent points out of
eight.  This version refuses to proceed silently in that situation.

Use --auto-heights for a guaranteed-disjoint, log-spaced ladder.
--------------------------------------------------------------------------

Usage
-----
    # 8 disjoint log-spaced heights, run serially
    python fig3_alpha_vs_height.py --auto-heights 8 --n-windows 24

    # one height per process (shell-level parallelism), then merge
    python fig3_alpha_vs_height.py --auto-heights 8 --only 3 \
        --cache ../data/processed/fig3_h3.npz --no-fig
    python fig3_alpha_vs_height.py --merge "../data/processed/fig3_h*.npz"

    # replot from a cache
    python fig3_alpha_vs_height.py --plot-only
"""

from __future__ import annotations

import argparse
import glob as _glob
import sys
from pathlib import Path

import numpy as np

import riemann_siegel as rs
from generate_sequence import (SequenceSpec, map_to_lattice,
                               occupations_from_sites, M10B, M11B, build)
from structure_factor_check import structure_factor, fit_alpha

A_NM, ELL_NM = 0.2504, 1.25
GUE_SPACING_STD = 0.41797            # asymptotic GUE value


# --------------------------------------------------------------------------
# Window bookkeeping
# --------------------------------------------------------------------------

def zeros_per_window(n_sites: int) -> int:
    """Zeros consumed to fill one `n_sites` lattice."""
    return int(2.2 * n_sites * A_NM / ELL_NM) + 64


def span_per_point(n_sites: int, n_windows: int) -> int:
    """Total index range consumed by one height point."""
    return n_windows * zeros_per_window(n_sites)


def auto_heights(n_points: int, n_sites: int, n_windows: int,
                 pad: float = 1.15, decades: float = 3.0) -> list[int]:
    """
    Log-spaced starting indices, guaranteed disjoint: consecutive starts are
    separated by at least `pad` times the span consumed by one point.
    """
    span = span_per_point(n_sites, n_windows)
    lo = int(pad * span)
    hi = lo * 10 ** decades
    grid = np.logspace(np.log10(lo), np.log10(hi), max(n_points - 1, 1))
    out = [0]
    for x in grid:
        out.append(max(int(round(x)), int(out[-1] + pad * span)))
    return out[:n_points]


def check_overlap(heights, n_sites: int, n_windows: int, strict: bool = True):
    """Report -- and by default refuse -- overlapping height windows."""
    span = span_per_point(n_sites, n_windows)
    h = sorted(int(x) for x in heights)
    bad = []
    for i in range(len(h) - 1):
        gap = h[i + 1] - h[i]
        if gap < span:
            bad.append((h[i], h[i + 1], 100.0 * (span - gap) / span))
    if bad:
        lines = [f"each height point consumes {span} zeros "
                 f"({n_windows} windows x {zeros_per_window(n_sites)}); "
                 f"these starts overlap:"]
        for lo, hi, pct in bad:
            lines.append(f"    n0={lo} and n0={hi} share {pct:.0f}% of their zeros")
        lines.append("Overlapping points are not statistically independent and "
                     "will bias any fitted trend.")
        lines.append("Use --auto-heights, widen --heights, or pass "
                     "--allow-overlap to proceed anyway.")
        text = "\n".join(lines)
        if strict:
            raise SystemExit("ERROR: " + text)
        print("WARNING: " + text, file=sys.stderr)
    return span, bad


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def alpha_for_height(n0: int, n_sites: int, n_windows: int) -> dict:
    spec = SequenceSpec("zeta", n_sites, A_NM, ELL_NM)
    x = np.arange(n_sites) * A_NM
    k0 = 2 * np.pi / ELL_NM
    n_pts = zeros_per_window(n_sites)

    acc, stds = None, []
    g_first = g_last = None
    for w in range(n_windows):
        g = rs.zeros(n_pts, start_index=n0 + w * n_pts)
        u = rs.unfold(g)
        stds.append(np.diff(u).std())
        if w == 0:
            g_first = float(g[0])
        g_last = float(g[-1])

        occ = occupations_from_sites(map_to_lattice(u, spec), n_sites)
        m = np.where(occ, M10B, M11B)
        eps = m / m.mean() - 1.0
        k, s = structure_factor(eps, x, taper="hann", k0=k0)
        acc = s / np.mean(eps ** 2) if acc is None else acc + s / np.mean(eps ** 2)

    a, e = fit_alpha(k, acc / n_windows, k0)
    return dict(n0=int(n0), alpha=a, alpha_err=e, gamma_first=g_first,
                gamma_last=g_last, gamma_geom=float(np.sqrt(g_first * g_last)),
                spacing_std=float(np.mean(stds)))


def gue_reference(n_sites: int, n_seeds: int) -> tuple[float, float]:
    acc = None
    k0 = 2 * np.pi / ELL_NM
    for seed in range(n_seeds):
        d = build(SequenceSpec("gue", n_sites, A_NM, ELL_NM, seed))
        k, s = structure_factor(d["eps"], d["x_nm"], taper="hann", k0=k0)
        acc = s / d["g2"] if acc is None else acc + s / d["g2"]
    return fit_alpha(k, acc / n_seeds, k0)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def weighted_trend(alpha, err, gamma) -> dict:
    """
    Weighted least squares of alpha against log10(gamma) with the UNSCALED
    covariance, which is the correct choice when `err` are true 1-sigma
    uncertainties.

    numpy.polyfit(cov=True) rescales the covariance by the residual variance
    and therefore understates the slope error here.  Do not use it for this
    test.  Significance is cross-checked against the likelihood-ratio
    Delta chi^2, which must agree.
    """
    a, e = np.asarray(alpha, float), np.asarray(err, float)
    x = np.log10(np.asarray(gamma, float))
    W = np.diag(1.0 / e ** 2)
    X = np.vstack([x, np.ones_like(x)]).T
    C = np.linalg.inv(X.T @ W @ X)
    b = C @ X.T @ W @ a
    slope, slope_err = float(b[0]), float(np.sqrt(C[0, 0]))

    r = a - X @ b
    chi2_line = float(r @ W @ r)
    wmean = float(np.average(a, weights=1.0 / e ** 2))
    wmean_err = float(1.0 / np.sqrt(np.sum(1.0 / e ** 2)))
    chi2_const = float(np.sum((a - wmean) ** 2 / e ** 2))

    return dict(slope=slope, slope_err=slope_err,
                sigma_slope=abs(slope) / slope_err if slope_err else float("nan"),
                sigma_dchi2=float(np.sqrt(max(chi2_const - chi2_line, 0.0))),
                chi2_const=chi2_const, chi2_line=chi2_line,
                dof_const=len(a) - 1, dof_line=len(a) - 2,
                wmean=wmean, wmean_err=wmean_err,
                sigma_above_one=(wmean - 1) / wmean_err,
                cross_alpha1=float((1 - b[1]) / b[0]) if b[0] else float("nan"))


def report(res: dict, label: str) -> None:
    print(f"\n{label}")
    print(f"  slope vs log10(gamma) = {res['slope']:+.3f} +/- {res['slope_err']:.3f}"
          f"   -> {res['sigma_slope']:.1f} sigma")
    print(f"  Delta chi2 cross-check= {res['sigma_dchi2']:.1f} sigma  "
          f"(const {res['chi2_const']:.2f}/{res['dof_const']}dof, "
          f"line {res['chi2_line']:.2f}/{res['dof_line']}dof)")
    print(f"  weighted mean alpha   = {res['wmean']:.3f} +/- {res['wmean_err']:.3f}"
          f"   ({res['sigma_above_one']:+.1f} sigma from alpha=1)")
    print(f"  extrapolated alpha=1  at gamma ~ 1e{res['cross_alpha1']:.1f}")
    ratio = res["chi2_const"] / res["dof_const"]
    print("  VERDICT: ", end="")
    if res["sigma_slope"] > 3 and ratio > 2:
        print("height dependence ESTABLISHED. alpha is a usable design knob.")
    else:
        print(f"height dependence NOT established (constant alpha still fits, "
              f"chi2/dof = {ratio:.2f}). Do not claim a tunable knob.")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _load_records(files):
    recs, meta = [], {}
    for f in files:
        d = np.load(f)
        n = np.atleast_1d(d["alpha"]).size
        for i in range(n):
            recs.append({k: float(np.atleast_1d(d[k])[i])
                         for k in ("n0", "alpha", "alpha_err",
                                   "gamma_geom", "spacing_std")})
        meta = dict(n_sites=int(d["n_sites"]), n_windows=int(d["n_windows"]),
                    alpha_gue=float(d["alpha_gue"]),
                    alpha_gue_err=float(d["alpha_gue_err"]))
    recs.sort(key=lambda r: r["n0"])
    return recs, meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-sites", type=int, default=12000)
    p.add_argument("--n-windows", type=int, default=24)
    p.add_argument("--heights", type=int, nargs="+", default=None,
                   help="explicit starting indices (checked for overlap)")
    p.add_argument("--auto-heights", type=int, default=None, metavar="N",
                   help="generate N guaranteed-disjoint log-spaced heights")
    p.add_argument("--only", type=int, default=None, metavar="I",
                   help="run only the I-th height of the ladder (0-based), "
                        "for shell-level parallelism")
    p.add_argument("--allow-overlap", action="store_true")
    p.add_argument("--cache", type=Path,
                   default=Path("../data/processed/fig3_alpha_vs_height.npz"))
    p.add_argument("--fig", type=Path,
                   default=Path("../figures/fig3_alpha_vs_height.pdf"))
    p.add_argument("--no-fig", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--merge", type=str, default=None, metavar="GLOB",
                   help="merge per-height caches, run the statistics, replot")
    p.add_argument("--estimate", action="store_true",
                   help="time one window per rung and project the total "
                        "runtime, then exit without measuring")
    args = p.parse_args()

    # ---------------- merge ----------------
    if args.merge:
        files = sorted(_glob.glob(args.merge))
        if not files:
            raise SystemExit(f"no caches matched {args.merge}")
        recs, meta = _load_records(files)
        n0 = np.array([r["n0"] for r in recs])
        al = np.array([r["alpha"] for r in recs])
        er = np.array([r["alpha_err"] for r in recs])
        gg = np.array([r["gamma_geom"] for r in recs])
        sd = np.array([r["spacing_std"] for r in recs])

        span, bad = check_overlap(n0, meta["n_sites"], meta["n_windows"],
                                  strict=False)
        print(f"\nmerged {len(recs)} height points from {len(files)} cache(s)")
        print(f"{'n0':>10}{'gamma_geom':>14}{'alpha':>18}{'sigma_s':>11}")
        for r in recs:
            print(f"{int(r['n0']):>10}{r['gamma_geom']:14.0f}"
                  f"{r['alpha']:11.3f} +/-{r['alpha_err']:.3f}"
                  f"{r['spacing_std']:11.4f}")
        print(f"{'GUE ref':>10}{'--':>14}{meta['alpha_gue']:11.3f} "
              f"+/-{meta['alpha_gue_err']:.3f}{GUE_SPACING_STD:11.4f}")
        print(f"\nsigma_s monotonic in all {len(sd)} points: "
              f"{bool(np.all(np.diff(sd) > 0))}")

        report(weighted_trend(al, er, gg),
               "ALL POINTS" + (" (OVERLAPPING -- upper bound only)" if bad else ""))
        if bad:
            keep = [0]
            for i in range(1, len(n0)):
                if n0[i] - n0[keep[-1]] >= span:
                    keep.append(i)
            keep = np.array(keep)
            if keep.size >= 3:
                report(weighted_trend(al[keep], er[keep], gg[keep]),
                       f"DISJOINT SUBSET (n={keep.size}) -- quote this one")
        _plot(gg, al, er, sd, meta["alpha_gue"], meta["alpha_gue_err"],
              args.fig, args.no_fig)
        return

    # ---------------- replot ----------------
    if args.plot_only:
        d = np.load(args.cache)
        _plot(d["gamma_geom"], d["alpha"], d["alpha_err"], d["spacing_std"],
              float(d["alpha_gue"]), float(d["alpha_gue_err"]),
              args.fig, args.no_fig)
        return

    # ---------------- measure ----------------
    if args.auto_heights:
        heights = auto_heights(args.auto_heights, args.n_sites, args.n_windows)
        print(f"auto ladder: {args.auto_heights} disjoint points, "
              f"{span_per_point(args.n_sites, args.n_windows)} zeros each")
        print("  " + ", ".join(map(str, heights)))
    elif args.heights:
        heights = args.heights
    else:
        heights = auto_heights(8, args.n_sites, args.n_windows)
        print("no --heights given; using the auto ladder:")
        print("  " + ", ".join(map(str, heights)))

    if args.only is not None:
        heights = [heights[args.only]]
    else:
        check_overlap(heights, args.n_sites, args.n_windows,
                      strict=not args.allow_overlap)

    if args.estimate:
        _estimate(heights, args.n_sites, args.n_windows)
        return

    print(f"\n{'n0':>10}{'gamma_0':>12}{'gamma_max':>12}"
          f"{'gamma_geom':>13}{'alpha':>17}{'sigma_s':>11}")
    recs = []
    for n0 in heights:
        r = alpha_for_height(int(n0), args.n_sites, args.n_windows)
        recs.append(r)
        print(f"{r['n0']:>10}{r['gamma_first']:12.0f}{r['gamma_last']:12.0f}"
              f"{r['gamma_geom']:13.0f}{r['alpha']:10.3f} +/-{r['alpha_err']:.3f}"
              f"{r['spacing_std']:11.4f}", flush=True)

    a_gue, e_gue = gue_reference(args.n_sites, args.n_windows)
    print(f"{'GUE ref':>10}{'--':>12}{'--':>12}{'--':>13}"
          f"{a_gue:10.3f} +/-{e_gue:.3f}{GUE_SPACING_STD:11.4f}")

    arr = lambda k: np.array([r[k] for r in recs])
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.cache, n0=arr("n0"), alpha=arr("alpha"),
             alpha_err=arr("alpha_err"), gamma_first=arr("gamma_first"),
             gamma_last=arr("gamma_last"), gamma_geom=arr("gamma_geom"),
             spacing_std=arr("spacing_std"), alpha_gue=a_gue,
             alpha_gue_err=e_gue, n_sites=args.n_sites,
             n_windows=args.n_windows)
    print(f"cache -> {args.cache}")

    if len(recs) >= 3:
        report(weighted_trend(arr("alpha"), arr("alpha_err"), arr("gamma_geom")),
               "TREND TEST")
    _plot(arr("gamma_geom"), arr("alpha"), arr("alpha_err"), arr("spacing_std"),
          a_gue, e_gue, args.fig, args.no_fig)


def _estimate(heights, n_sites: int, n_windows: int) -> None:
    """
    Time a single window at each rung and project the total.

    Cost is dominated by the Riemann-Siegel main sum, whose length is
    floor(sqrt(t/2pi)), so runtime grows as sqrt(gamma): the top rungs of a
    log-spaced ladder dominate. Run this before launching a long job.
    """
    import time
    n_pts = zeros_per_window(n_sites)
    print(f"\n{'n0':>12}{'gamma':>13}{'N_terms':>9}{'s/window':>11}{'rung est':>12}")
    total = 0.0
    for n0 in heights:
        t0 = time.time()
        g = rs.zeros(n_pts, start_index=int(n0))
        dt = time.time() - t0
        rung = dt * n_windows
        total += rung
        print(f"{int(n0):>12}{g[0]:13.0f}{int((g[0]/(2*np.pi))**0.5):9d}"
              f"{dt:11.2f}{rung/60:9.1f} min", flush=True)
    print(f"\n  serial total   : {total/60:.0f} min")
    print(f"  parallel       : the slowest rung governs")
    print(f"  NOTE: the GUE reference adds ~{n_windows * 0.5 / 60:.1f} min per process.")


def _plot(gamma, alpha, err, sd, a_gue, e_gue, path: Path, skip: bool) -> None:
    if skip:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax[0].errorbar(gamma, alpha, yerr=err, fmt="o-", ms=4, lw=1.2, capsize=2.5,
                   color="C0", label="Riemann zeros")
    ax[0].axhspan(a_gue - e_gue, a_gue + e_gue, color="C1", alpha=0.25)
    ax[0].axhline(a_gue, color="C1", lw=1.1, ls="--", label="GUE surrogate")
    ax[0].axhline(1.0, color="k", lw=0.8, ls=":", label=r"class II ($\alpha=1$)")
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"sampled height $\bar\gamma$")
    ax[0].set_ylabel(r"$\alpha$")
    ax[0].legend(fontsize=7, frameon=False)

    ax[1].plot(gamma, sd, "s-", ms=4, lw=1.2, color="C0")
    ax[1].axhline(GUE_SPACING_STD, color="C1", lw=1.1, ls="--")
    ax[1].text(0.97, 0.10, "GUE", transform=ax[1].transAxes, ha="right",
               fontsize=8, color="C1")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"sampled height $\bar\gamma$")
    ax[1].set_ylabel(r"spacing std $\sigma_s$")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    print(f"figure -> {path}")


if __name__ == "__main__":
    main()
