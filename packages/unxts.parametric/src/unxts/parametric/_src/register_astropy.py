"""Astropy interop for ``ParametricQuantity`` (registered on import).

Registered only when ``astropy`` is importable; a no-op otherwise so that
``import unxts.parametric`` never requires the optional ``astropy`` dependency.
"""

__all__: tuple[str, ...] = ()

try:
    from astropy.units import Quantity as AstropyQuantity
except ImportError:  # pragma: no cover - astropy is an optional dependency
    pass
else:
    from plum import conversion_method

    import unxt_api as uapi
    from .parametric import ParametricQuantity

    @conversion_method(type_from=AstropyQuantity, type_to=ParametricQuantity)  # type: ignore[arg-type]
    def convert_astropy_quantity_to_parametric_quantity(
        q: AstropyQuantity, /
    ) -> ParametricQuantity:
        """Convert an `astropy.units.Quantity` to a `ParametricQuantity`.

        Examples
        --------
        >>> from astropy.units import Quantity as AstropyQuantity
        >>> from plum import convert
        >>> from unxts.parametric import ParametricQuantity

        >>> convert(AstropyQuantity(1.0, "cm"), ParametricQuantity)
        ParametricQuantity(Array(1., dtype=float32), unit='cm')

        """
        u = uapi.unit_of(q)
        return ParametricQuantity(uapi.ustrip(u, q), u)
