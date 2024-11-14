# pylint: disable=import-error

"""unxt: Quantities in JAX.

Copyright (c) 2023 Galactic Dynamics. All rights reserved.
"""

__all__: list[str] = []

from typing import TypeAlias

import astropy.units as apyu

Unit: TypeAlias = apyu.Unit | apyu.UnitBase | apyu.CompositeUnit
