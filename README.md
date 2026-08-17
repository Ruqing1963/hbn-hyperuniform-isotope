# Hyperuniform isotope engineering of phonon transport in hexagonal boron nitride

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![tests](https://github.com/Ruqing1963/hbn-hyperuniform-isotope/actions/workflows/ci.yml/badge.svg)](https://github.com/Ruqing1963/hbn-hyperuniform-isotope/actions)

Code, data, and manuscript source for *Hyperuniform isotope engineering of
phonon transport: exact scaling laws and a stealthy transparency window in
hexagonal boron nitride*.

---

## Abstract

Within the Born–Oppenheimer approximation, interatomic force constants are
invariant under isotopic substitution, so an isotope pattern perturbs the
dynamical matrix through the mass factors alone. This makes the *spatial
arrangement* of isotopes a design variable independent of the isotopic
variance `g2` that has been the sole focus of isotope engineering to date.

We generalise the Tamura theory of phonon–isotope scattering to spatially
correlated arrangements by introducing the mass-fluctuation structure factor
`S_eps(k) = g2 * s(k)`. When `s(k) ~ (k/k0)^alpha` at small wavenumber, the
relaxation rate obeys the exact scaling law

```
1/tau_iso  ∝  omega^(d + 1 + alpha)
```

in `d` dimensions, with closed-form angular prefactor `2/(alpha+2)`. **The
hyperuniformity exponent contributes an additive, dimension-independent
correction to the Rayleigh exponent**, orthogonal to the dimensional baseline.

Two consequences are developed:

- **Class II (`alpha = 1`).** Mapping the unfolded nontrivial zeros of the
  Riemann zeta function — whose pair statistics belong to the GUE universality
  class — onto a 1D `10B`/`11B` modulation converts the Rayleigh `omega^4`
  into `omega^5`.
- **Class I (stealthy).** For arrangements with `S_eps(k) = 0` on a finite
  interval `0 < k < K`, the golden-rule rate vanishes *identically*, not
  asymptotically, for all `omega < vK/2`. This is a hard phonon transparency
  window rather than a power-law suppression.

The proposed observable is a differential transmission `Delta_T(omega)`
between composition-matched samples, which cancels the Casimir and anharmonic
backgrounds to first order. At liquid-helium temperature this opens a **4–7
THz** window with a predicted ~20% signal, at the cost of a fabrication
specification (**1.25 nm** pitch over **300 μm**) that exceeds current growth
capability. The paper is positioned as a theory and design-specification
study; that gap is stated explicitly rather than concealed.

---

## What this repository does and does not contain

**Contains.** Sequence generation and lattice mapping; the structure-factor
convergence check that validates `alpha = 1` survives discretisation; a
transfer-matrix transport module (Lyapunov exponents and Landauer
transmission); the manuscript source; a regression test suite.

**Does not contain.** The primitive-cell DFT force constants. Stage 0
(Sec. III A of the paper) requires an external DFT code; `code/transport.py`
ships with a sound-velocity-matched surrogate force constant so that the
pipeline runs end to end without one. Absolute magnitudes in Table I of the
paper require the real force constants and the full transverse channel sum.

**On the Riemann zeros — read this before assuming `alpha = 1`.**
Every prediction depends on the arrangement only through `alpha`. The zeros
are the canonical generator of GUE (`beta = 2`) statistics, but **only
asymptotically in height**. Sequences built from zeros at computationally
accessible heights are measurably *more rigid* than GUE:

Finite-height departures from GUE are themselves well known (semiclassical
prime-pair corrections to the form factor; Odlyzko needed zeros near 1e20 for
clean agreement). What `fig3_alpha_vs_height.py` settles is their size, and
whether they are tunable. Eight **disjoint** windows of 24x5352 zeros each,
sampled heights 1.2e3 to 6.2e7:

| gamma_bar | 1.2e3 | 1.4e5 | 3.4e5 | 8.9e5 | 2.5e6 | 7.2e6 | 2.1e7 | 6.2e7 | GUE |
|---|---|---|---|---|---|---|---|---|---|
| alpha | 2.02 | 1.47 | 1.22 | 1.29 | 1.32 | 1.47 | 1.32 | 1.39 | 0.973 ± 0.029 |
| sigma_s | .4028 | .4069 | .4091 | .4108 | .4120 | .4140 | .4146 | .4161 | .4180 |

(individual alpha uncertainties 0.27–0.40)

**The result is a clean positive plus a clean negative.**

- *Established.* The realizable zeta sequence is more rigid than GUE:
  `alpha = 1.49 ± 0.12`, which is 4.1 sigma above 1. By Eq. (12) this deepens
  the suppression from `omega^5` to about `omega^5.4`.
- *Established.* `sigma_s` rises monotonically in all eight windows
  (+0.0029/decade), extrapolating to the GUE value near `gamma ~ 1e8.4`.
- **Ruled out.** Height is *not* a tunable knob for `alpha`. The fitted slope
  is `−0.138 ± 0.075` (1.8 sigma; the Delta-chi2 cross-check agrees) and a
  constant fits well (`chi2/dof = 0.73`). Dropping the single lowest window
  leaves the remaining seven flat: slope `+0.013 ± 0.149` (0.1 sigma),
  `chi2/dof = 0.07` about `alpha = 1.364 ± 0.135`. A preliminary run over five
  decades gave slope `−0.234`; extending to eight decades *reduced* it to
  `−0.138` with unchanged significance, which is how a spurious trend behaves,
  not an unresolved one.

That `sigma_s` drifts while `alpha` does not is itself the interesting part:
the short-range spacing statistic and the `k -> 0` asymptotics converge toward
GUE at different rates.

> **If you re-run this, do not use overlapping windows.** Each height point
> consumes `n_windows * n_pts` consecutive zeros. The first attempt used
> starts 0/5000/20000/60000, which shared 96/88/69/30 percent of their zeros
> and produced a trend that was partly an artefact of the overlap. The script
> now refuses such a ladder; use `--auto-heights N`.

Use `--source gue` when a clean `alpha = 1` reference is wanted;
`--source zeta --start-index N` selects the height. The default `--source gue` uses a
Dumitriu–Edelman tridiagonal surrogate, which is statistically equivalent at
the level of pair correlations, requires no external data, and is orders of
magnitude faster. Use `--source zeta` to reproduce with the actual zeros
(via `mpmath`, or an Odlyzko table).

---

## Requirements

| Package | Version | Required for |
|---|---|---|
| Python | ≥ 3.10 | everything |
| NumPy | ≥ 1.24 | everything |
| SciPy | ≥ 1.10 | GUE spectra, unfolding |
| Matplotlib | ≥ 3.7 | figures |
| mpmath | ≥ 1.3 | optional cross-check of the zero finder |
| phonopy | ≥ 2.20 | DFT force-constant stage only |
| pytest | any | test suite |

```bash
git clone https://github.com/Ruqing1963/hbn-hyperuniform-isotope.git
cd hbn-hyperuniform-isotope
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or:  conda env create -f environment.yml && conda activate hyperuniform-isotope
```

Manuscript compilation additionally needs a TeX distribution with REVTeX 4.2
(`texlive-publishers` on Debian/Ubuntu).

---

## Reproducibility instructions

Everything below runs on a laptop. Full pipeline: `make all`.

### Step 1 — Generate the arrangements

```bash
cd code
python generate_sequence.py --all --n-sites 200000
```

Produces `data/processed/seq_{gue,random,periodic,stealthy}.npz`. All four are
forced to **identical isotopic composition** — this is the load-bearing
constraint of the differential measurement (Sec. IV B), not a cosmetic detail,
and a warning is raised if the generator misses the target by more than 1e-4.

Expected output:

```
target 10B concentration c = 0.199833  (from class II mapping)
  gue       c=0.199833  g2=1.358378e-03
  random    c=0.199833  g2=1.358378e-03
  periodic  c=0.199833  g2=1.358378e-03
  stealthy  c=0.199833  g2=1.358378e-03
```

### Step 2 — Structure-factor convergence check (Fig. 1)

```bash
python structure_factor_check.py
```

Produces `figures/fig1_structure_factor.pdf` and
`data/processed/alpha_table.csv`. This is the check that answers the obvious
referee objection — *does mapping onto a discrete lattice destroy
`alpha = 1`?* Reference output at `n_sites = 60000`:

| arrangement | c | g2 | alpha (Hann) | alpha (rect) |
|---|---|---|---|---|
| gue | 0.19983 | 1.3584e-03 | **1.043 ± 0.059** | 0.992 ± 0.041 |
| random | 0.19983 | 1.3584e-03 | 0.016 ± 0.047 | 0.018 ± 0.042 |
| periodic | 0.19983 | 1.3584e-03 | −0.017 ± 0.047 | −0.011 ± 0.032 |
| stealthy | 0.19983 | 1.3584e-03 | 0.086 ± 0.067 | 0.031 ± 0.041 |

The script also prints the band-edge contamination bound, Eq. (21):
`pi^2/3 * (a/ell)^2 = 0.1320`, against the exact `1 - sinc^2(k0 a/2) = 0.1252`
— agreement to 5% at the design point `a/ell = 0.2`.

> **Three caveats you must read before quoting any fitted alpha.**
>
> 1. **Estimator variance.** A raw periodogram ordinate is exponentially
>    distributed: its standard deviation equals its mean. A single realisation
>    on a sparse `k` grid gives fitted errors of ±0.5 and is worthless. The
>    script evaluates every Fourier mode by FFT and band-averages in
>    logarithmically spaced bins (Bartlett smoothing); without this the
>    apparent exponent drifts to 0.65–0.89 and *looks* like degraded
>    hyperuniformity when it is not. Windowing choice matters most at the
>    smallest `k`; both tapered and untapered values are reported so the size
>    of the artefact is visible.
> 2. **Composition matching injects `alpha = 0`.** Flipping a fraction `f` of
>    sites at random to hit a target concentration superposes a white
>    component of relative size `~ f / [c(1-c)]` on `s(k)`. On a class I
>    sequence, `f ~ 1e-3` is already enough to measure `alpha ~ 0`. Every
>    generator must hit the composition *by construction*.
> 3. **Class I arrangements are floor-limited here.** The `periodic` and
>    `stealthy` rows measure `alpha ~ 0` not because they lack long-range
>    order but because a log–log slope is the wrong diagnostic once `s(k)`
>    reaches the discretisation floor of Eq. (18). For class I, report `s(k)`
>    at the band edge directly. The `stealthy` default annealing schedule
>    (4000 swaps) is a demo setting and gives only ~2× suppression; production
>    runs need ~10^2 more iterations.

### Step 3 — Differential transmission (Fig. 2)

```bash
python transport.py
```

Produces `figures/fig2_differential_transmission.pdf`.

> **Convergence warning.** `Delta_T` is a small difference of two O(1)
> quantities. With the default 8 configurations and a 4000-site chain,
> configuration-to-configuration fluctuations (±3–5%) exceed the systematic
> signal and `Delta_T` changes sign between frequency points. Converging the
> difference needs O(10^3) configurations. **The Lyapunov route
> (`transport.lyapunov`) is far better conditioned** and is the recommended
> primary estimator of the exponent in Eq. (12), as stated in Sec. III E of
> the paper. The transmission demo is included to show the pipeline, not as a
> converged result.

### Step 4 — Tests

```bash
pytest -q tests
```

Five regression tests: `alpha = 1` for the class II sequence, `alpha = 0` for
the random control, `g2` against its analytic value, exactness of composition
matching, and Eq. (21) against the exact `1 - sinc^2`.

### Step 5 — Figure 3 (alpha vs height)

```bash
python fig3_alpha_vs_height.py              # a few minutes; caches to .npz
python fig3_alpha_vs_height.py --plot-only  # replot from cache
```

### Step 6 — Manuscript

```bash
make paper      # real REVTeX 4.2 build; needs texlive-publishers
make preview    # fallback PDF without REVTeX -- NOT for submission
make zh         # Chinese reading version (xelatex + Noto CJK fonts)
```

`paper/paper_zh.tex` is a complete Chinese translation for reading and
internal review. Every equation number, figure number, reference number and
numerical value matches the English source, so the two can be read side by
side. **The English version is authoritative for submission.** Building it
needs `xelatex` and the Noto Serif/Sans CJK SC fonts
(`fonts-noto-cjk` on Debian/Ubuntu); no `ctex` or `xeCJK` package is required,
since line breaking is handled by XeTeX's own `\XeTeXlinebreaklocale`.

`paper/paper_preview.pdf` is committed for convenience. It is built by
`paper/_mkpreview.py` + `revtex_shim.tex`, an `article`-class emulation, for
environments without REVTeX 4.2. **It is not the REVTeX typesetting**: line
breaking, float placement, and the bibliography all differ, and the page count
will not match. Regenerate with `make paper` before submitting anywhere.

Two REVTeX behaviours have no article equivalent and are rewritten by the
generator rather than worked around in TeX, because both fail *silently*:

- `\author` accumulates in REVTeX; in `article` the last one wins and every
  earlier co-author is dropped without a warning. All author/affiliation
  groups are folded into one block with numbered affiliations.
- REVTeX places `\begin{abstract}` before `\maketitle`; `article` typesets it
  where it appears, i.e. above the title. The block is moved after
  `\maketitle`.

---

## Repository structure

```
.
├── README.md
├── LICENSE                          MIT
├── CITATION.cff                     citation metadata (GitHub + Zenodo)
├── .zenodo.json                     Zenodo deposition metadata
├── Makefile                         `make all` runs the full pipeline
├── requirements.txt / environment.yml
├── code/
│   ├── generate_sequence.py         GUE/zeta/random/periodic/stealthy →
│   │                                lattice mapping [Eq. (17)], composition
│   │                                matching [Sec. IV B]
│   ├── structure_factor_check.py    FFT + band-averaged s(k); alpha fits;
│   │                                discretisation check [Eqs. (18)–(21)]
│   └── transport.py                 transfer matrices: Lyapunov exponent,
│                                    Landauer transmission, Delta_T
├── data/
│   ├── raw/                         cached zeta ordinates (regenerated)
│   └── processed/                   seq_*.npz, alpha_table.csv
├── figures/
│   ├── fig1_structure_factor.pdf    s(k) and the windowing artefact
│   └── fig2_differential_transmission.pdf
├── paper/
│   ├── paper.tex                    REVTeX 4.2 manuscript
│   └── refs.bib
├── tests/test_pipeline.py
└── .github/workflows/ci.yml
```

---

## Citation

```bibtex
@article{hyperuniform_isotope_2026,
  title  = {Hyperuniform isotope engineering of phonon transport: exact
            scaling laws and a stealthy transparency window in hexagonal
            boron nitride},
  author = {Wu, Ronghua and Chen, Ruqing},
  journal= {arXiv preprint arXiv:XXXX.XXXXX},
  year   = {2026}
}

@software{hyperuniform_isotope_code_2026,
  title    = {Code and data for hyperuniform isotope engineering of phonon
              transport in h-BN},
  author   = {Wu, Ronghua and Chen, Ruqing},
  year     = {2026},
  doi      = {10.5281/zenodo.XXXXXXX},
  publisher= {Zenodo}
}
```

## License

MIT (see `LICENSE`). If you reuse the sequence-generation or structure-factor
code, please also cite the paper.
