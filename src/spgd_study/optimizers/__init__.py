"""Saddle-point-escaping optimisers compared in this study.

- SPGD: every Iter_P steps, sample N_P uniform perturbations, accept the
        candidate with the lowest loss if it is not worse than current.
- RPGD: same generation cost as SPGD, but selects the kept candidate
        UNIFORMLY AT RANDOM instead of by argmin. Isolates SPGD's
        steepest-selection rule (vs. the perturbation alone).
- PGD : single perturbation every Iter_P steps, no selection, no
        acceptance check — closer in spirit to Jin et al. (2017).

All three follow the same *forward-only closure* protocol; see SPGD docstring.
"""

from .pgd import PGD
from .rpgd import RPGD
from .spgd import SPGD

__all__ = ["SPGD", "PGD", "RPGD"]
