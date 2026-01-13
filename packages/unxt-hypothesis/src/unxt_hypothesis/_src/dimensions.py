"""Hypothesis strategies for named dimensions.

Derived from Astropy physical types list at
https://github.com/astropy/astropy/blob/dc7afddeb21316baa9f5d71f6d32c35fd02ca3db/astropy/units/physical.py.
The names are captured here as a finalized tuple and filtered to those that
`unxt.dimension` can resolve.
"""

__all__ = ("DIMENSION_NAMES", "named_dimensions")

from typing import Final

from hypothesis import strategies as st

import unxt as u

# The Astropy physical type names that map cleanly to unxt dimensions.
DIMENSION_NAMES: Final[tuple[str, ...]] = (
    "absement",
    "absity",
    "acceleration",
    "action",
    "amount of substance",
    "angle",
    "angular acceleration",
    "angular frequency",
    "angular momentum",
    "angular speed",
    "angular velocity",
    "area",
    "bandwidth",
    "catalytic activity",
    "chemical potential",
    "column density",
    "compressibility",
    "crackle",
    "data quantity",
    "diffusivity",
    "dimensionless",
    "dose of ionizing radiation",
    "dynamic viscosity",
    "electrical capacitance",
    "electrical charge",
    "electrical charge density",
    "electrical conductance",
    "electrical conductivity",
    "electrical current",
    "electrical current density",
    "electrical dipole moment",
    "electrical field strength",
    "electrical flux density",
    "electrical impedance",
    "electrical mobility",
    "electrical potential",
    "electrical reactance",
    "electrical resistance",
    "electrical resistivity",
    "electromagnetic field strength",
    "electron density",
    "electron flux",
    "energy",
    "energy density",
    "energy flux",
    "entropy",
    "force",
    "frequency",
    "frequency drift",
    "heat capacity",
    "illuminance",
    "impulse",
    "inductance",
    "irradiance",
    "jerk",
    "jolt",
    "jounce",
    "kinematic viscosity",
    "length",
    "linear density",
    "luminance",
    "luminous efficacy",
    "luminous emittance",
    "luminous flux",
    "luminous intensity",
    "magnetic field strength",
    "magnetic flux",
    "magnetic flux density",
    "magnetic helicity",
    "magnetic moment",
    "magnetic reluctance",
    "mass",
    "mass attenuation coefficient",
    "mass density",
    "mass flux",
    "molality",
    "molar concentration",
    "molar conductivity",
    "molar heat capacity",
    "molar volume",
    "moment of inertia",
    "momentum",
    "momentum density",
    "number density",
    "opacity",
    "particle flux",
    "permeability",
    "permittivity",
    "photon flux",
    "photon flux density",
    "photon flux density wav",
    "photon surface brightness",
    "photon surface brightness wav",
    "plate scale",
    "polarization density",
    "pop",
    "pounce",
    "power",
    "power density",
    "pressure",
    "radiance",
    "radiant flux",
    "radiant intensity",
    "reaction rate",
    "snap",
    "solid angle",
    "specific energy",
    "specific entropy",
    "specific heat capacity",
    "specific volume",
    "spectral flux density",
    "spectral flux density wav",
    "speed",
    "stress",
    "surface brightness",
    "surface brightness wav",
    "surface charge density",
    "surface mass density",
    "surface tension",
    "temperature",
    "temperature gradient",
    "thermal conductance",
    "thermal conductivity",
    "thermal resistance",
    "thermal resistivity",
    "time",
    "torque",
    "velocity",
    "volume",
    "volumetric flow rate",
    "volumetric rate",
    "wavenumber",
    "work",
    "yank",
)


@st.composite
def named_dimensions(draw: st.DrawFn) -> u.AbstractDimension:
    """Generate a named dimension from Astropy's physical type catalogue.

    This strategy samples from a finalized, pre-computed set of 134 named
    physical dimensions derived from Astropy's physical type system. These
    dimensions are guaranteed to be resolvable by `unxt.dimension()`.

    The set of dimension names is curated from Astropy's physical types and
    excludes names that cannot be directly interpreted by unxt (e.g., names
    with parenthetical qualifiers like "electrical charge (EMU)").

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function (automatically provided by @st.composite).

    Returns
    -------
    unxt.AbstractDimension
        A randomly selected dimension from the set of named dimensions.

    Examples
    --------
    Basic usage - generate a dimension:

    >>> from hypothesis import given
    >>> import unxt as u
    >>> import unxt_hypothesis as ust

    >>> @given(dim=ust.named_dimensions())
    ... def test_physics_dimension(dim):
    ...     assert isinstance(dim, u.AbstractDimension)

    Using with unit generation to create quantities of any dimension:

    >>> @given(
    ...     dim=ust.named_dimensions(),
    ...     q=ust.quantities(u.dimension("length"), shape=3),
    ... )
    ... def test_dimension_and_quantity(dim, q):
    ...     # dim varies across test runs from 134 physical types
    ...     assert isinstance(dim, u.AbstractDimension)
    ...     assert q.shape == (3,)

    Combine with units strategy:

    >>> @given(unit=ust.units(ust.named_dimensions()))
    ... def test_any_unit(unit):
    ...     # Generate units from any named physical dimension
    ...     assert isinstance(unit, u.AbstractUnit)
    ...     assert u.dimension_of(unit) in [
    ...         u.dimension(name) for name in ust.DIMENSION_NAMES
    ...     ]

    Property-based testing across all 134 physical types:

    >>> @given(dim=ust.named_dimensions())
    ... def test_all_dimensions_are_valid(dim):
    ...     # This test will run with 100 different random dimensions
    ...     # (by default, Hypothesis runs 100 examples)
    ...     assert isinstance(dim, u.AbstractDimension)
    ...     assert dim in {u.dimension(name) for name in ust.DIMENSION_NAMES}

    See Also
    --------
    DIMENSION_NAMES : The tuple of all available dimension names.
    unxt.dimension : Create a dimension from a name string.
    units : Generate units with a specific dimension.
    quantities : Generate quantities with a specific dimension.

    """
    name = draw(st.sampled_from(DIMENSION_NAMES))
    return u.dimension(name)
