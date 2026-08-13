"""Named units for the fully-determined natural unit systems.

`astropy.units` has no names for the Planck or Hartree atomic base units, so
they can only be spelled as a scale times an SI unit -- and
`astropy.units.UnitBase.to_string` truncates past six significant figures, so
that spelling neither reads well nor reparses:

    >>> import unxt as u
    >>> u.unitsystems.planck["length"].to_string()  # doctest: +SKIP
    '1.61626e-35 m'

Naming them fixes both at once: ``l_P`` is short, and it reparses exactly.

Each unit is *defined from the same constant expression* that builds its unit
system rather than from a hard-coded number, so the values track whatever
CODATA revision `astropy.constants` ships.

The units are registered with `astropy.units.add_enabled_units` at import, so
``unxt.unit("l_P")`` resolves. That mutates astropy's registry process-wide --
a deliberate trade for making the spelling round-trip. All eight names were
checked to be unclaimed; note that ``a0`` and ``Eh`` are *not* free (astropy
already parses them as dimensionless and as a time, respectively), which is
why the Bohr radius is ``a_0`` and there is no Hartree-energy unit here.

Upstreaming these into astropy would let unxt drop this module; see the
tracking issue.
"""

# ruff: noqa: N816
#    ``l_P``/``m_P``/``t_P``/``T_P`` are the standard physics symbols for the
#    Planck units; spelling them ``l_p`` would be wrong, not merely unidiomatic.

__all__ = ("ATOMIC_UNITS", "PLANCK_UNITS")

import astropy.units as apyu
import numpy as np
from astropy.constants import (  # pylint: disable=E0611
    G as const_G,  # noqa: N811
    Ryd as const_Ryd,
    a0 as const_a0,
    c as const_c,
    e as const_e,
    h as const_h,
    hbar as const_hbar,
    k_B as const_kB,
    m_e as const_me,
)

# ============================================================================
# Planck units: hbar = c = G = k_B = 1

#: Planck length.
l_P = apyu.def_unit(
    "l_P",
    np.sqrt(const_hbar * const_G / const_c**3).decompose(),
    format={"latex": r"\ell_\mathrm{P}"},
    doc="Planck length, sqrt(hbar G / c^3).",
)

#: Planck mass.
m_P = apyu.def_unit(
    "m_P",
    np.sqrt(const_hbar * const_c / const_G).decompose(),
    format={"latex": r"m_\mathrm{P}"},
    doc="Planck mass, sqrt(hbar c / G).",
)

#: Planck time.
t_P = apyu.def_unit(
    "t_P",
    np.sqrt(const_hbar * const_G / const_c**5).decompose(),
    format={"latex": r"t_\mathrm{P}"},
    doc="Planck time, sqrt(hbar G / c^5).",
)

#: Planck temperature.
T_P = apyu.def_unit(
    "T_P",
    (np.sqrt(const_hbar * const_c**5 / const_G) / const_kB).decompose(),
    format={"latex": r"T_\mathrm{P}"},
    doc="Planck temperature, sqrt(hbar c^5 / G) / k_B.",
)

#: The Planck base units, in the order `unxt.unitsystems.planck` declares them.
PLANCK_UNITS: tuple[apyu.UnitBase, ...] = (l_P, m_P, t_P, T_P)


# ============================================================================
# Atomic (Hartree) units: m_e = hbar = e = 4*pi*eps0 = 1

#: Bohr radius. Spelled ``a_0`` because astropy already parses ``a0`` as
#: dimensionless.
a_0 = apyu.def_unit(
    "a_0",
    const_a0.decompose(),
    # ``a_{0}`` not ``a_0``: astropy ignores a format string identical to the
    # unit's own name, and then escapes the underscore into ``a\_0``.
    format={"latex": r"a_{0}"},
    doc="Bohr radius, the atomic unit of length.",
)

#: Electron rest mass.
m_e = apyu.def_unit(
    "m_e",
    const_me.decompose(),
    format={"latex": r"m_\mathrm{e}"},
    doc="Electron rest mass, the atomic unit of mass.",
)

#: Atomic unit of time, ``hbar / E_h`` with ``E_h`` the Hartree energy.
#:
#: The Hartree energy is *twice* the Rydberg energy, not one Rydberg.
t_au = apyu.def_unit(
    "t_au",
    (const_hbar / (2 * const_Ryd * const_h * const_c)).decompose(),
    format={"latex": r"t_\mathrm{au}"},
    doc="Atomic unit of time, hbar / E_h.",
)

#: Elementary charge. A unit named ``e`` does not disturb scientific notation:
#: ``1e5 m`` still parses as a scale times a metre.
e = apyu.def_unit(
    "e",
    const_e.si.decompose(),
    format={"latex": r"e"},
    doc="Elementary charge, the atomic unit of charge.",
)

#: The atomic base units, in the order `unxt.unitsystems.atomic` declares them.
ATOMIC_UNITS: tuple[apyu.UnitBase, ...] = (a_0, m_e, t_au, e)


# ============================================================================

# Make the names parseable, so ``unxt.unit("l_P")`` resolves and a unit system's
# ``repr`` reconstructs. This is process-wide; see the module docstring.
apyu.add_enabled_units([*PLANCK_UNITS, *ATOMIC_UNITS])
