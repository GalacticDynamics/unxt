---
title: "unxt: A Python package for unit-aware computing with JAX"
tags:
  - Python
  - Numerical Computing
  - Scientific Computing
authors:
  - name: Nathaniel Starkman
    orcid: 0000-0003-3954-3291
    affiliation: "1"
    corresponding: true
  - name: Adrian M. Price-Whelan
    orcid: 0000-0003-0872-7098
    affiliation: "2"
  - name: Jake Nibauer
    orcid: 0000-0001-8042-5794
    affiliation: "3"
affiliations:
  - index: 1
    name:
      Brinson Prize Fellow at Kavli Institute for Astrophysics and Space
      Research, Massachusetts Institute of Technology, USA
    ror: 042nb2s44
  - index: 2
    name: Center for Computational Astrophysics, Flatiron Institute, USA
    ror: 00sekdz59
  - index: 3
    name: Department of Physics, Princeton University, USA
    ror: 00hx57361
date: 15 November 2024
bibliography: paper.bib
---

# Summary

`unxt` is a Python package for unit-aware computing with JAX [@jax:18], which is
a high-performance numerical computing library that enables automatic
differentiation and just-in-time compilation to accelerate code execution on
multiple compute architectures. `unxt` is built on top of `quax` [@quax:23],
which provides a framework for building array-like objects that can be used with
JAX. `unxt` extends `quax` to provide support for unit-aware computing using the
`astropy.units` package [@astropy:13; @astropy:22] as a units backend. `unxt`
provides seamless integration of physical units into high performance numerical
computations, significantly enhancing the capabilities of JAX for scientific
applications.

The primary purpose of `unxt` is to facilitate unit-aware computations in JAX,
ensuring that operations involving physical quantities are handled correctly and
consistently. This is crucial for avoiding errors in scientific calculations,
such as those that could lead to significant consequences like the infamous Mars
Climate Orbiter incident [@nasa:98]. `unxt` is designed to be intuitive, easy to
use, and performant, allowing for a straightforward implementation of units into
existing JAX codebases.

`unxt` is accessible to researchers and developers, providing a user-friendly
interface for defining and working with units and unit systems. It supports both
static and dynamic definitions of unit systems, allowing for flexibility in
various computational environments. Additionally, `unxt` leverages multiple
dispatch to enable deep interoperability with other libraries, currently
`astropy`, and to support custom array-like objects in JAX. This extensibility
makes `unxt` a powerful tool for a wide range of scientific and engineering
applications, where unit-aware computations are essential.

# Statement of Need

JAX is a powerful tool for high-performance numerical computing, offering
features such as automatic differentiation, just-in-time compilation, and
support for sharding computations across multiple devices. It excels in
providing unified interfaces to various compute architectures, including CPUs,
GPUs, and TPUs, to accelerate code execution [@jax:18]. However, JAX operates
primarily on "pure" arrays, which means it lacks support to define custom
array-like objects, including those that can handle units, and to use those use
those objects in within the JAX ecosystem. While JAX can handle PyTrees with
some pre-programmed support and the ability to register additional support, the
operations it performs are still fundamentally array-based. This limitation
poses a challenge for scientific applications that require handling of physical
units.

Astropy has been an invaluable resource for the scientific community, with over
10,000 citations to its initial paper and more than 2,000 citations to its 2022
paper [@astropy:13; @astropy:22]. One of the foundational sub-packages within
Astropy is `astropy.units`, which provides robust support for units and
quantities, enabling the propagation of units through NumPy functions. This
functionality ensures that scientific calculations involving physical quantities
are handled correctly and consistently. However, despite JAX's numpy-like API,
it does not support the same level of extensibility, and `astropy.units` cannot
be directly extended to work with JAX. This gap highlights the need for a
solution that integrates the powerful unit-handling capabilities of Astropy with
the high-performance computing features of JAX.

`unxt` addresses this gap by providing a function-oriented framework—consistent
with the style of JAX—for handling units and dimensions, with an object-oriented
front-end that will be familiar to users of `astropy.units`. By leveraging
`quax`, `unxt` defines a `Quantity` class that seamlessly integrates with JAX
functions. This integration is achieved by providing a comprehensive set of
overrides for JAX primitives, ensuring that users can utilize the `Quantity`
class without needing to worry about the underlying JAX interfacing. This design
allows users to perform unit-aware computations effortlessly, maintaining the
high performance and flexibility that JAX offers while ensuring the correctness
and consistency of operations involving physical quantities.

# Related Works

`unxt` is designed to be extensible to other unitful-computation libraries. The
`unxt` package is not intended to replace these libraries, but rather to provide
a JAX-optimized frontend. Some prominent libraries include:

- `astropy.units` [@astropy:13; @astropy:22]. The `unxt` package currently uses
  the unit conversion framework from `astropy.units` package in its backend,
  providing a more flexible front-end interface and particularly JAX-compatible
  Quantity classes for doing array computations with units.
- `unyt` [@unyt:2018]. The `unyt` library is a popular Python package for
  unit-aware computations. It provides `Quantity` classes that work with (at
  time of writing) `numpy` [@numpy:2020] and `dask` [@dask:2016] arrays.
- `pint` [@pint]. The `pint` library is a popular Python package for unit-aware
  computations. It provides `Quantity` classes that work with many array types,
  but not `jax` (at time of writing).

# Acknowledgements

Support for this work was provided by The Brinson Foundation through a Brinson
Prize Fellowship grant.

The authors thank the Astropy collaboration and many contributors for their work
on `astropy`, which has been invaluable to the scientific community. Members of
the `unxt` development team are also core developers and maintainers of the
`astropy.units` package, and we had `astropy.units` as our guiding star while
developing `unxt`. The authors also thank Dan Foreman-Mackey for useful
discussions, and the attendees of the 2024 JAXtronomy workshop at the Center for
Computational Astrophysics at the Flatiron Institute. We also extend our
gratitude to Patrick Kidger for his valuable communications and guidance on
using `quax` to ensure seamless integration of `unxt` with `jax`.

# References
