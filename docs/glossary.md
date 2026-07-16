# 📇 Glossary

```{glossary}
Dimension(s)
  A dimension refers to a measurable extent of a physical quantity, such as length, time, or mass.
  A set of dimensions are taken to be the "base" dimensions.
  "Derived" dimensions are those that can be expressed in terms of the fundamental dimensions. For example, taking length as a base dimension, area is a derived dimension that can be expressed as length squared.
  Dimensions express the nature of a quantity and are independent of the specific units used to measure them.

Unit(s)
  Units are measures of a dimension. Some common units are meters for length or seconds for time or Joules for energy. There are generally many units that can measure a single dimension.
  Units can belong to different systems (like SI or Imperial).

Unit System
  A unit system is a collection of units that are defined and used together for consistency and standardization.
  Examples include the International System of Units (SI), the Imperial system, and natural unit systems such as Planck units or atomic units. Each system specifies base units (like meters, kilograms) and derived units (like joules or newtons), allowing for the expression of Quantities in terms of these units.

USys
  A shorthand for "unit system", used in class names for concision.

Quantity
  A quantity refers to a property of a system that can be measured or calculated, expressed as a number combined with a unit.
  Examples include 5 meters, 10 seconds, or 50 joules.
  In `unxt`, {class}`~unxt.Quantity` (alias `u.Q`) is the **default**, lightweight
  quantity class. It is *non-parametric*: it tracks units without encoding the
  physical dimension in its type, so it stays a single class — and a single JAX
  pytree node type — for every dimension instead of proliferating one per
  dimension.

ParametricQuantity
  `ParametricQuantity` (alias `up.PQ`, from the separate
  [`unxts.parametric`](packages/unxts.parametric/index) package) is the opt-in,
  *parametric* quantity class. It encodes the physical dimension in its type
  (e.g. ``ParametricQuantity['length']``), enabling dimension-specific
  multiple-dispatch and runtime dimension checking, at the cost of a distinct
  class — and a distinct JAX pytree node type — per dimension. (This grows the
  type and dispatch surface; it does not add `jax.jit` compilations, which are
  driven by the static unit for either class.)

BareQuantity
  A **deprecated** alias of {class}`~unxt.Quantity` (`u.Q`); new code should use
  `Quantity`. See the {ref}`migration guide <migration-v2>`.

Multiple-Dispatch
  Multiple dispatch is a programming paradigm in which the method or function to be called is determined by the runtime types of the arguments.
  This allows for  flexible and extensible code, as different implementations can be provided for different types of arguments.
  See [Wikipedia](https://en.wikipedia.org/wiki/Multiple_dispatch) for more information.

Parametric Class
  A parametric class is a class that is defined by a set of parameters. The `unxts.parametric.ParametricQuantity` class is an example of a parametric class, as it is defined by its dimensions -- ``ParametricQuantity['length']`` is a quantity with dimensions of length, while ``ParametricQuantity['time']`` is a quantity with dimensions of time.

Non-parametric Class
  A non-parametric class is a class that is not defined by a set of parameters.
  The `unxt.quantity.Quantity` class is an example of a non-parametric class; it does not use any parameter.

Equality
  For quantities, `==`. On a `StaticValue`-backed quantity it is **unit-blind** — two quantities are equal only when their unit *labels* match (so they remain distinct `jax.jit` `static_argnames` keys). Array-backed quantities compare element-wise and unit-aware. See {term}`Equivalence` for a cross-unit comparison.

Equivalence
  Whether two quantities are the **same physical amount**, accounting for unit conversion — e.g. `1000 m` and `1 km` are equivalent though not equal. Exposed as {func}`unxt.equivalent` and the `is_equivalent` method. The same `equivalent` function also reports whether two {term}`Unit System`s span the same dimensions.

```
