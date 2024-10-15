# Conventions

Things I want to say:

- Functional versus object-oriented programming
- JAX is primarily functional
- Python is normally object-oriented
- As a JAX library it's important to support functional
- But unit-vendoring libraries are object-oriented and so users are familiar
  with those APIs
- So we need to strike a balance between the two.
- We can do this by providing both functional and object-oriented APIs.
- The functional APIs are the primary APIs.
- But the object-oriented APIs are easy to use and call the functional APIs, so
  lose none of the power.

Also, JAX only supports JAX arrays. We use `quax` to support custom array-ish
objects. Among many things, `quax` uses multiple dispatch to enable custom
array-ish objects, like `Quantity`. We then use multiple dispatch to enable the
functional APIs and lots of interoperability.

Examples

```{code-block} python

>>> from unxt import Quantity, uconvert

>>> q = Quantity(1, 'm')
>>> q
Quantity['length'](Array(1, dtype=int32, weak_type=True), unit='m')

>>> q.to('cm')
Quantity['length'](Array(100., dtype=float32, weak_type=True), unit='cm')

>>> uconvert("cm", q)
Quantity['length'](Array(100., dtype=float32, weak_type=True), unit='cm')

```
