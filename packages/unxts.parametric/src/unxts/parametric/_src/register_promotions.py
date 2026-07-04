"""Promotion rules involving ParametricQuantity (registered on import)."""

from plum import add_promotion_rule

from unxt.quantity import AbstractAngle, StaticQuantity

from .parametric import ParametricQuantity

add_promotion_rule(AbstractAngle, ParametricQuantity, ParametricQuantity)
add_promotion_rule(StaticQuantity, ParametricQuantity, ParametricQuantity)
