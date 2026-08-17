# Raw inputs

`zeta_zeros.npy` — cached ordinates of the nontrivial zeros of zeta(s),
generated on first use by `generate_sequence.py --source zeta` via `mpmath`.
Not tracked in git; regenerated automatically, or replace with an Odlyzko
table and load it with `generate_sequence.load_zeta_table`.

Nothing else here is required to reproduce the paper: the default
`--source gue` surrogate needs no external data.
