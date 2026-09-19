"""RASPA helper utilities — unit-cell expansion for minimum-image rule.

The key function ``get_unit_cell`` reads a CIF file and returns a UnitCells
triple ``"nx ny nz"`` such that every box dimension (accounting for
non-orthogonal lattice angles, e.g. hexagonal γ=120°) exceeds
``2 × cutoff`` — the RASPA minimum-image convention.

This module is imported lazily by ``agents.registry._exec_henry`` and
``agents.slurm`` so it must remain side-effect-free on import.
"""

import math
import re
from pathlib import Path


def _parse_cif_cell(cif_path: str):
    """Extract lattice parameters (a, b, c, alpha, beta, gamma) from a CIF.

    Returns (a, b, c, alpha_deg, beta_deg, gamma_deg) in Å / degrees,
    or *None* if parsing fails.
    """
    try:
        text = Path(cif_path).read_text(errors="ignore")
    except Exception:
        return None

    def _get(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    a = _get(r"_cell_length_a\s+([\d.]+)")
    b = _get(r"_cell_length_b\s+([\d.]+)")
    c = _get(r"_cell_length_c\s+([\d.]+)")
    alpha = _get(r"_cell_angle_alpha\s+([\d.]+)")
    beta = _get(r"_cell_angle_beta\s+([\d.]+)")
    gamma = _get(r"_cell_angle_gamma\s+([\d.]+)")

    if any(v is None for v in (a, b, c)):
        return None

    alpha = alpha or 90.0
    beta = beta or 90.0
    gamma = gamma or 90.0
    return a, b, c, alpha, beta, gamma


def _effective_lengths(a, b, c, alpha_deg, beta_deg, gamma_deg):
    """Compute effective Cartesian box lengths for one unit cell.

    For a general triclinic cell the *minimum perpendicular distance*
    between opposite faces (which is what RASPA uses for the minimum-image
    rule along each lattice-vector direction) is:

        d_a = V / (b·c·sin(α))          — perpendicular to **a**
        d_b = V / (a·c·sin(β))          — perpendicular to **b**
        d_c = V / (a·b·sin(γ))          — perpendicular to **c**

    where V is the unit-cell volume:

        V = a·b·c·sqrt(1 - cos²α - cos²β - cos²γ + 2·cosα·cosβ·cosγ)

    For *orthorhombic* cells (α=β=γ=90°) this simplifies to d_a=a, etc.

    Parameters
    ----------
    a, b, c : float   lattice lengths in Å
    alpha, beta, gamma : float  lattice angles in degrees

    Returns
    -------
    (d_a, d_b, d_c) : effective perpendicular box lengths in Å
    """
    alpha_r = math.radians(alpha_deg)
    beta_r = math.radians(beta_deg)
    gamma_r = math.radians(gamma_deg)

    ca, cb, cg = math.cos(alpha_r), math.cos(beta_r), math.cos(gamma_r)
    sa, sb, sg = math.sin(alpha_r), math.sin(beta_r), math.sin(gamma_r)

    volume = a * b * c * math.sqrt(max(1e-12, 1 - ca**2 - cb**2 - cg**2 + 2*ca*cb*cg))

    # Avoid division by zero for degenerate cells
    d_a = volume / (b * c * sa) if (b * c * sa) > 1e-12 else a
    d_b = volume / (a * c * sb) if (a * c * sb) > 1e-12 else b
    d_c = volume / (a * b * sg) if (a * b * sg) > 1e-12 else c

    return d_a, d_b, d_c


def get_unit_cell(cif_path: str, cutoff: float = 12.0,
                  max_product: int = 64) -> str:
    """Return a ``"nx ny nz"`` string satisfying the minimum-image rule.

    The algorithm:
    1. Parse lattice parameters from *cif_path*.
    2. Compute effective perpendicular lengths per unit cell.
    3. Find the smallest integers (nx, ny, nz) such that
       nx·d_a ≥ 2·cutoff, ny·d_b ≥ 2·cutoff, nz·d_c ≥ 2·cutoff.
    4. If the product nx·ny·nz exceeds *max_product*, relax to the
       smallest product that still satisfies the constraint (may exceed
       max_product for very anisotropic cells but warns).

    Returns
    -------
    str  e.g. ``"2 2 4"``
    """
    cell = _parse_cif_cell(cif_path)
    if cell is None:
        # Cannot parse CIF — return safe default
        return "2 2 2"

    a, b, c, alpha, beta, gamma = cell
    d_a, d_b, d_c = _effective_lengths(a, b, c, alpha, beta, gamma)

    min_box = 2.0 * cutoff  # RASPA minimum-image requirement

    def _ceil_div(length, target):
        """Smallest integer n such that n * length >= target."""
        if length <= 0:
            return 2  # safety fallback
        return max(1, math.ceil(target / length + 1e-9))

    nx = _ceil_div(d_a, min_box)
    ny = _ceil_div(d_b, min_box)
    nz = _ceil_div(d_c, min_box)

    # Sanity cap: warn if product is very large but still return valid triple
    product = nx * ny * nz
    if product > max_product:
        import sys
        print(
            f"  ⚠️ get_unit_cell: product {nx}×{ny}×{nz}={product} exceeds "
            f"recommended max {max_product}. Consider using a larger cutoff or "
            f"checking the CIF structure.",
            file=sys.stderr, flush=True,
        )

    return f"{nx} {ny} {nz}"
