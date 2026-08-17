#!/usr/bin/env python3
"""
generate_sequence.py
====================

Generate isotope occupation sequences for 1D mass modulation of h-BN and map
them onto the discrete boron sublattice.

Five arrangement classes are supported, all at *identical* isotopic
composition (this is the load-bearing constraint of the differential
measurement, Sec. IV B of the paper):

    random      alpha = 0    (control)
    gue         alpha = 1    (class II surrogate; GUE beta=2 spectrum)
    zeta        alpha = 1    (class II; unfolded Riemann zeta zeros)
    periodic    alpha -> inf (class I)
    stealthy    S(k) = 0 for k < K  (class I, strong)

On the physics: every prediction of the paper depends on the arrangement only
through the small-k exponent alpha [Eq. (12)].  The Riemann zeros enter as an
exactly characterised generator of GUE (beta = 2) statistics, nothing more.
The 'gue' surrogate is statistically equivalent at the level of pair
correlations and is the default because it requires no external data and is
orders of magnitude faster.  Use source='zeta' to reproduce with the actual
zeros.

Usage
-----
    python generate_sequence.py --source gue --n-sites 200000 --out ../data/processed/seq_gue.npz
    python generate_sequence.py --all --out-dir ../data/processed

Author: [ ]
License: MIT
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from scipy.optimize import minimize_scalar

# --------------------------------------------------------------------------
# Physical constants for h-BN (Sec. IV E of the paper)
# --------------------------------------------------------------------------

A_LATTICE_NM = 0.2504      # in-plane lattice constant a  [nm]
ELL_DESIGN_NM = 1.25       # design pitch (= 5a, Eq. (21))  [nm]
M10B = 10.012937           # amu
M11B = 11.009305           # amu


# --------------------------------------------------------------------------
# Class II generators
# --------------------------------------------------------------------------

def gue_spectrum(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Eigenvalues of a GUE (beta = 2) matrix via the Dumitriu-Edelman
    tridiagonal model.  O(n^2) instead of O(n^3), and no dense matrix.
    """
    diag = rng.normal(0.0, np.sqrt(2.0), n)
    k = np.arange(n - 1, 0, -1)
    offd = np.sqrt(rng.chisquare(2.0 * k))
    return np.sort(eigvalsh_tridiagonal(diag / np.sqrt(2.0), offd / np.sqrt(2.0)))


def unfold_semicircle(ev: np.ndarray, bulk: float = 0.15) -> np.ndarray:
    """
    Unfold a GUE spectrum to unit mean spacing using the exact semicircle
    counting function, with the radius refined by least squares against the
    empirical staircase.

    Only the bulk fraction (1 - 2*bulk) of the spectrum is returned: the edges
    obey Tracy-Widom rather than bulk sine-kernel statistics and would
    contaminate the small-k structure factor.
    """
    n = ev.size

    def counting(x, R):
        x = np.clip(x, -R, R)
        return n * (0.5 + (x * np.sqrt(np.maximum(R ** 2 - x ** 2, 0.0)) / R ** 2
                           + np.arcsin(x / R)) / np.pi)

    idx = np.arange(1, n + 1) - 0.5
    lo, hi = int(0.1 * n), int(0.9 * n)
    R0 = float(np.max(np.abs(ev)))
    R = minimize_scalar(
        lambda R: np.sum((counting(ev[lo:hi], R) - idx[lo:hi]) ** 2),
        bracket=(0.9 * R0, 1.1 * R0),
    ).x
    u = counting(ev, R)
    return np.sort(u[int(bulk * n): int((1 - bulk) * n)])


def zeta_zeros_unfolded(n: int, cache: Path | None = None) -> np.ndarray:
    """
    First n nontrivial zeros of zeta(s) on the critical line, unfolded to unit
    mean spacing by the Riemann-von Mangoldt density [Eq. (16)]:

        gamma_tilde_n = (gamma_n / 2pi) * log(gamma_n / 2pi e)

    Requires mpmath.  This is slow (seconds per thousand zeros); results are
    cached.  For n >~ 1e5 use Odlyzko's tables instead and load them with
    `load_zeta_table`.
    """
    if cache is not None and cache.exists():
        g = np.load(cache)
        if g.size >= n:
            return _unfold_rvm(g[:n])

    from mpmath import zetazero, mp
    mp.dps = 20
    g = np.array([float(zetazero(k).imag) for k in range(1, n + 1)])
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, g)
    return _unfold_rvm(g)


def load_zeta_table(path: Path, n: int | None = None) -> np.ndarray:
    """Load an Odlyzko-format table of zero ordinates (one per line)."""
    g = np.loadtxt(path)
    return _unfold_rvm(g if n is None else g[:n])


def _unfold_rvm(gamma: np.ndarray) -> np.ndarray:
    u = (gamma / (2 * np.pi)) * np.log(gamma / (2 * np.pi * np.e))
    return np.sort(u)


# --------------------------------------------------------------------------
# Lattice mapping
# --------------------------------------------------------------------------

@dataclass
class SequenceSpec:
    source: str
    n_sites: int
    a_nm: float = A_LATTICE_NM
    ell_nm: float = ELL_DESIGN_NM
    seed: int = 0
    stealth_chi: float = 0.45
    stealth_iters: int = 4000
    start_index: int = 0        # zeta only: height of the first zero used


def map_to_lattice(u_unit: np.ndarray, spec: SequenceSpec) -> np.ndarray:
    """
    Map a unit-mean-spacing sequence onto the discrete sublattice [Eq. (17)]:

        j_n = round(ell * u_n / a)

    Returns *integer site indices*, deduplicated and sorted.  Collisions
    (two points rounding onto the same site) are the discretisation channel
    quantified by Eq. (18)-(21); they are dropped, not merged, which is the
    conservative choice.
    """
    j = np.round(spec.ell_nm * (u_unit - u_unit[0]) / spec.a_nm).astype(np.int64)
    j = np.unique(j)
    return j[(j >= 0) & (j < spec.n_sites)]


def occupations_from_sites(sites: np.ndarray, n_sites: int) -> np.ndarray:
    """Boolean occupation array: True = 10B, False = 11B."""
    occ = np.zeros(n_sites, dtype=bool)
    occ[sites] = True
    return occ


def match_composition(occ: np.ndarray, target_c: float,
                      rng: np.random.Generator,
                      warn_frac: float = 1e-4) -> np.ndarray:
    """
    Force the 10B fraction to exactly `target_c` by flipping the minimum number
    of randomly chosen sites.

    This is not cosmetic.  Sec. IV B: a composition mismatch delta_c between
    the class II sample and its random control produces a spurious difference
    in g2 that mimics the signal.  All sequences compared in a single run MUST
    pass through this function with the same target_c.

    WARNING -- random flips inject an alpha = 0 component
    -----------------------------------------------------
    Flipping a fraction f of sites at random superposes an uncorrelated
    (white) contribution of relative size ~ f / [c(1-c)] on s(k).  On a
    hyperuniform sequence this *destroys* the small-k suppression: a periodic
    arrangement (class I, s -> 0 below the first Bragg peak) reduced to
    f ~ 1e-3 random flips already measures alpha ~ 0 rather than alpha >> 1.

    Every generator must therefore hit the target composition BY
    CONSTRUCTION, leaving this routine only single-digit residuals.  A warning
    is emitted when the flip fraction exceeds `warn_frac`.
    """
    occ = occ.copy()
    n = occ.size
    n_target = int(round(target_c * n))
    n_have = int(occ.sum())
    if abs(n_have - n_target) > warn_frac * n:
        import warnings
        warnings.warn(
            f"match_composition flipping {abs(n_have-n_target)}/{n} sites "
            f"({abs(n_have-n_target)/n:.2e}); this injects a white (alpha=0) "
            f"component into s(k). Fix the generator instead.", RuntimeWarning)
    if n_have > n_target:
        on = np.flatnonzero(occ)
        occ[rng.choice(on, n_have - n_target, replace=False)] = False
    elif n_have < n_target:
        off = np.flatnonzero(~occ)
        occ[rng.choice(off, n_target - n_have, replace=False)] = True
    return occ


# --------------------------------------------------------------------------
# Control arrangements
# --------------------------------------------------------------------------

def random_occupation(n_sites: int, c: float, rng) -> np.ndarray:
    occ = np.zeros(n_sites, dtype=bool)
    occ[rng.choice(n_sites, int(round(c * n_sites)), replace=False)] = True
    return occ


def periodic_occupation(n_sites: int, c: float) -> np.ndarray:
    """
    Exactly `round(c * n_sites)` sites, equally spaced.  The count is fixed by
    construction so that match_composition() has nothing to do -- see the
    warning in its docstring.
    """
    m = int(round(c * n_sites))
    occ = np.zeros(n_sites, dtype=bool)
    occ[np.unique(np.round(np.arange(m) * n_sites / m).astype(int) % n_sites)] = True
    return occ


def stealthy_occupation(n_sites: int, c: float, chi: float,
                        rng, n_iter: int = 4000) -> np.ndarray:
    """
    Collective-coordinate construction of a stealthy binary arrangement:
    minimise sum_{0<k<K} |sum_n eps_n exp(i k x_n)|^2 by simulated-annealing
    swaps of 10B/11B labels.

    K = chi * k0 with k0 = 2 pi / ell.  For a binary mass constraint in 1D,
    chi <~ 0.5 is the practical ceiling (Sec. II E).
    """
    occ = random_occupation(n_sites, c, rng)
    eps = np.where(occ, 1.0 - c, -c)

    n_k = max(4, int(chi * c * n_sites / 2))
    kk = 2 * np.pi * np.arange(1, n_k + 1) / n_sites   # (n_k,)

    # rho_m = sum_n eps_n exp(i k_m n); basis columns are generated on the fly
    # (O(n_k) memory instead of O(n_k * n_sites) -- the dense basis is ~GB)
    rho = np.exp(1j * np.outer(kk, np.arange(n_sites))) @ eps if n_sites <= 4096 \
        else _rho_fft(eps, n_k)
    cost = float(np.sum(np.abs(rho) ** 2))
    T0, T1 = cost / n_k * 0.1, cost / n_k * 1e-6

    for it in range(n_iter):
        T = T0 * (T1 / T0) ** (it / max(n_iter - 1, 1))
        i = int(rng.choice(np.flatnonzero(occ)))
        j = int(rng.choice(np.flatnonzero(~occ)))
        d = np.exp(1j * kk * j) - np.exp(1j * kk * i)
        rho_new = rho + d
        cost_new = float(np.sum(np.abs(rho_new) ** 2))
        if cost_new < cost or rng.random() < np.exp(-(cost_new - cost) / max(T, 1e-30)):
            occ[i], occ[j] = False, True
            rho, cost = rho_new, cost_new
    return occ


def _rho_fft(eps: np.ndarray, n_k: int) -> np.ndarray:
    """Collective coordinates at the first n_k Fourier modes, via one FFT."""
    return np.fft.fft(eps)[1:n_k + 1].conj()


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def build(spec: SequenceSpec, target_c: float | None = None) -> dict:
    rng = np.random.default_rng(spec.seed)
    n = spec.n_sites

    if spec.source in ("gue", "zeta"):
        # oversample: unfolding discards the spectral edges, rounding drops
        # collisions, so ask for ~2.2x the number of points we need
        n_pts = int(2.2 * n * spec.a_nm / spec.ell_nm) + 64
        if spec.source == "gue":
            u = unfold_semicircle(gue_spectrum(n_pts, rng))
        else:
            # Riemann-Siegel + Gram bracketing: ~1.5e4 zeros/s, vs ~8/s for
            # mpmath.zetazero.  NOTE: alpha depends on start_index -- low-lying
            # zeros are more rigid than GUE (see README).
            import riemann_siegel as rs
            u = rs.unfold(rs.zeros(n_pts, start_index=spec.start_index))
        sites = map_to_lattice(u, spec)
        occ = occupations_from_sites(sites, n)
        c_nat = occ.mean()
    else:
        c_nat = spec.a_nm / spec.ell_nm
        if spec.source == "random":
            occ = random_occupation(n, c_nat, rng)
        elif spec.source == "periodic":
            occ = periodic_occupation(n, c_nat)
        elif spec.source == "stealthy":
            occ = stealthy_occupation(n, c_nat, spec.stealth_chi, rng,
                                      spec.stealth_iters)
        else:
            raise ValueError(f"unknown source: {spec.source}")

    if target_c is not None:
        occ = match_composition(occ, target_c, rng)

    c = float(occ.mean())
    masses = np.where(occ, M10B, M11B)
    mbar = float(masses.mean())
    eps = masses / mbar - 1.0
    g2 = float(np.mean(eps ** 2))

    return dict(
        occupation=occ, eps=eps, masses=masses,
        concentration=c, m_bar=mbar, g2=g2,
        x_nm=np.arange(n) * spec.a_nm,
        meta=json.dumps(asdict(spec)),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="gue",
                   choices=["gue", "zeta", "random", "periodic", "stealthy"])
    p.add_argument("--all", action="store_true",
                   help="generate every arrangement at matched composition")
    p.add_argument("--n-sites", type=int, default=200_000)
    p.add_argument("--ell-nm", type=float, default=ELL_DESIGN_NM)
    p.add_argument("--a-nm", type=float, default=A_LATTICE_NM)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start-index", type=int, default=0,
                   help="zeta only: index of the first zero (controls height, "
                        "and therefore alpha -- see README)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("../data/processed"))
    args = p.parse_args()

    sources = ["gue", "random", "periodic", "stealthy"] if args.all else [args.source]

    # composition is fixed by the reference (class II) arrangement, then
    # imposed on every control -- see match_composition()
    ref = build(SequenceSpec("gue", args.n_sites, args.a_nm, args.ell_nm, args.seed,
                             start_index=args.start_index))
    target_c = ref["concentration"]
    print(f"target 10B concentration c = {target_c:.6f}  (from class II mapping)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for src in sources:
        spec = SequenceSpec(src, args.n_sites, args.a_nm, args.ell_nm, args.seed,
                            start_index=args.start_index)
        out = build(spec, target_c=target_c)
        path = args.out or (args.out_dir / f"seq_{src}.npz")
        np.savez_compressed(path, **out)
        print(f"  {src:9s} c={out['concentration']:.6f}  g2={out['g2']:.6e}  -> {path}")


if __name__ == "__main__":
    main()
