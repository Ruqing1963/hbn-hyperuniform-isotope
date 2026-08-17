#!/usr/bin/env python3
"""
transport.py
============

Phonon transport through a 1D mass-modulated chain by transfer matrices.
No large matrix is ever diagonalised (Sec. III E of the paper).

Model
-----
Harmonic chain, nearest-neighbour force constant K, site masses M_n:

    -M_n omega^2 u_n = K (u_{n+1} - 2 u_n + u_{n-1})

    => (u_{n+1}, u_n)^T = T_n (u_n, u_{n-1})^T,
       T_n = [[2 - M_n omega^2 / K, -1], [1, 0]]

The force constant K is isotope-independent (Born-Oppenheimer, Sec. II A) and
is taken from the primitive-cell DFT calculation; only the masses change
between arrangements.  This is what makes arbitrarily long aperiodic
sequences tractable.

Observables
-----------
lyapunov(...)      inverse localisation length gamma(omega); O(N), the
                   cleanest route to the exponent d+1+alpha of Eq. (12)
transmission(...)  Landauer transmission through a finite chain embedded in
                   leads of the mean mass
differential(...)  Delta_T(omega) = T_classII(omega) - T_random(omega),
                   the proposed observable of Sec. IV

Usage
-----
    python transport.py --in-dir ../data/processed \
                        --fig ../figures/fig2_differential_transmission.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# h-BN parameters used for the frequency axis (Sec. IV E)
V_SOUND_NM_PER_PS = 20.0        # in-plane LA velocity, 2e4 m/s = 20 nm/ps


def force_constant_from_velocity(m_bar_amu: float, a_nm: float,
                                 v_nm_per_ps: float) -> float:
    """
    K such that the long-wavelength chain velocity a*sqrt(K/M) matches v.
    Units: amu / ps^2.  (Replace with the DFT value of Phi for production runs;
    this is the sound-velocity-matched surrogate used for the demo.)
    """
    return m_bar_amu * (v_nm_per_ps / a_nm) ** 2


def lyapunov(masses: np.ndarray, K: float, omegas: np.ndarray,
             reorth_every: int = 16) -> np.ndarray:
    """
    Smallest Lyapunov exponent gamma(omega) = 1/xi(omega) from the transfer
    matrix product, with periodic renormalisation to prevent overflow.

    Cost O(N) per frequency, no diagonalisation.
    """
    n = masses.size
    out = np.empty(omegas.size)
    for i, w in enumerate(omegas):
        d = 2.0 - masses * (w * w) / K
        v = np.array([1.0, 0.0])
        acc = 0.0
        for j in range(n):
            v = np.array([d[j] * v[0] - v[1], v[0]])
            if j % reorth_every == 0:
                nrm = np.hypot(v[0], v[1])
                if nrm > 0:
                    acc += np.log(nrm)
                    v /= nrm
        nrm = np.hypot(v[0], v[1])
        if nrm > 0:
            acc += np.log(nrm)
        out[i] = acc / n
    return out


def _transfer_product(d: np.ndarray) -> np.ndarray:
    """Full 2x2 transfer matrix product (no renormalisation; use for short chains)."""
    M = np.eye(2)
    for dj in d:
        M = np.array([[dj, -1.0], [1.0, 0.0]]) @ M
    return M


def transmission(masses: np.ndarray, K: float, omegas: np.ndarray,
                 m_lead: float) -> np.ndarray:
    """
    Landauer transmission |t|^2 of a finite chain between semi-infinite leads
    of uniform mass `m_lead`.

    Lead dispersion: omega = 2 sqrt(K/m_lead) |sin(q/2)|; propagating for
    omega < 2 sqrt(K/m_lead).
    """
    wmax = 2.0 * np.sqrt(K / m_lead)
    out = np.zeros(omegas.size)
    for i, w in enumerate(omegas):
        if w <= 0 or w >= wmax:
            continue
        q = 2.0 * np.arcsin(w / wmax)          # lead wavevector
        M = _transfer_product(2.0 - masses * (w * w) / K)
        e = np.exp(1j * q)
        # match u_n = e^{iqn} + r e^{-iqn} (left) to t e^{iqn} (right)
        A = np.array([[e, np.conj(e)], [1.0, 1.0]])
        b = M @ np.array([1.0, np.conj(e)])
        # solve for (t, r): outgoing amplitude ratio
        t = (b[0] - np.conj(e) * b[1]) / (e - np.conj(e))
        r = (e * b[1] - b[0]) / (e - np.conj(e))
        denom = abs(t) ** 2 + abs(r) ** 2
        out[i] = abs(t) ** 2 / denom if denom > 0 else 0.0
    return np.clip(out, 0.0, 1.0)


def load(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    return dict(masses=d["masses"], m_bar=float(d["m_bar"]),
                g2=float(d["g2"]), c=float(d["concentration"]),
                a_nm=meta["a_nm"], ell_nm=meta["ell_nm"], name=path.stem)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=Path("../data/processed"))
    p.add_argument("--fig", type=Path,
                   default=Path("../figures/fig2_differential_transmission.pdf"))
    p.add_argument("--n-chain", type=int, default=4000,
                   help="chain length used for the transmission demo")
    p.add_argument("--n-omega", type=int, default=80)
    p.add_argument("--n-config", type=int, default=8)
    args = p.parse_args()

    ref = load(args.in_dir / "seq_gue.npz")
    rnd = load(args.in_dir / "seq_random.npz")
    if abs(ref["c"] - rnd["c"]) > 2e-3:
        raise SystemExit("composition mismatch > 0.2%: differential invalid (Sec. IV B)")

    K = force_constant_from_velocity(ref["m_bar"], ref["a_nm"], V_SOUND_NM_PER_PS)
    wmax = 2.0 * np.sqrt(K / ref["m_bar"])
    omegas = np.linspace(0.02, 0.45, args.n_omega) * wmax

    # frequency axis in THz:  f = omega / 2pi, omega in 1/ps
    f_thz = omegas / (2 * np.pi)

    N = args.n_chain
    Tg = np.zeros_like(omegas)
    Tr = np.zeros_like(omegas)
    rng = np.random.default_rng(0)
    for _ in range(args.n_config):
        o = int(rng.integers(0, ref["masses"].size - N))
        Tg += transmission(ref["masses"][o:o + N], K, omegas, ref["m_bar"])
        o = int(rng.integers(0, rnd["masses"].size - N))
        Tr += transmission(rnd["masses"][o:o + N], K, omegas, rnd["m_bar"])
    Tg /= args.n_config
    Tr /= args.n_config
    dT = Tg - Tr

    print(f"{'f (THz)':>10}{'T_gue':>10}{'T_rand':>10}{'Delta_T':>12}")
    for i in range(0, omegas.size, max(1, omegas.size // 12)):
        print(f"{f_thz[i]:10.3f}{Tg[i]:10.4f}{Tr[i]:10.4f}{dT[i]:12.5f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax[0].plot(f_thz, Tg, lw=1.3, label="class II (GUE)")
    ax[0].plot(f_thz, Tr, lw=1.3, label="random alloy")
    ax[0].set_xlabel("f (THz)"); ax[0].set_ylabel(r"$\mathcal{T}$"); ax[0].legend(fontsize=8)
    ax[1].plot(f_thz, dT, lw=1.3, color="C3")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel("f (THz)"); ax[1].set_ylabel(r"$\Delta\mathcal{T}$")
    ax[1].set_title("differential observable")
    fig.tight_layout()
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=200)
    print(f"\nfigure -> {args.fig}")
    print("NOTE: this 1D demo uses a sound-velocity-matched force constant and a "
          "4000-site chain.\nIt reproduces the sign and shape of Delta_T, not the "
          "absolute magnitudes of Table I,\nwhich require the DFT force constants "
          "and the full k_parallel channel sum (Sec. III D).")


if __name__ == "__main__":
    main()
