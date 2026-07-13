"""Promotion rules involving ParametricQuantity (registered on import)."""

from plum import add_promotion_rule

from .parametric import ParametricQuantity
from unxt.quantity import AbstractAngle, StaticQuantity

add_promotion_rule(AbstractAngle, ParametricQuantity, ParametricQuantity)
add_promotion_rule(StaticQuantity, ParametricQuantity, ParametricQuantity)
