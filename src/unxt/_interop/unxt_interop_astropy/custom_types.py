"""Custom type annotations for astropy."""

__all__ = ("APYUnits",)


from typing import TypeAlias

import astropy.units as apyu

APYUnits: TypeAlias = (
    apyu.UnitBase | apyu.Unit | apyu.FunctionUnitBase | apyu.StructuredUnit
)
