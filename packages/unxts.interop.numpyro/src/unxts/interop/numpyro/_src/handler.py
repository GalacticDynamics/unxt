"""The `unitful` NumPyro handler."""

__all__ = ("unitful",)

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import jax
import numpy as np
import numpyro.distributions as dist
import numpyro.primitives as prim
from numpyro import handlers
from numpyro.contrib.control_flow.cond import cond_wrapper
from numpyro.contrib.control_flow.scan import scan_wrapper
from numpyro.distributions.util import lazy_property
from numpyro.primitives import Messenger
from plum import NotFoundLookupError

import unxt as u
from .rules import (
    UnitfulError,
    _dimensioned,
    _has_quantity,
    _is_q,
    carrying_params,
    leaf_path,
    site_dimension,
    strip_powers,
    unwrap,
)

# Key in `msg["infer"]` holding ``(usys, unit)`` of the `unitful` that ran a
# sample site, with the unit the site was stripped in. `replay` copies a guide's
# `infer` into the model's message.
_TAG_KEY = "_unitful_site"


def _check_container(name: str, v: Any) -> None:
    """Raise if `v` is a container that holds Quantities."""
    if not _is_q(v) and any(_is_q(x) for x in jax.tree.leaves(v, is_leaf=_is_q)):
        err = (
            f"Site '{name}': the value is a container that holds Quantities. "
            "unitful strips a value only when it is a Quantity itself; give "
            "each Quantity a site of its own, or pass one Quantity array."
        )
        raise UnitfulError(err)


def _dimensionless(name: str, what: str, v: Any) -> Any:
    """Strip a dimensionless Quantity; raise if `v` has a dimension."""
    if u.dimension_of(v) != u.dimension("dimensionless"):
        err = (
            f"Site '{name}': {what} has dimension {u.dimension_of(v)}, "
            "expected dimensionless."
        )
        raise UnitfulError(err)
    return u.ustrip("", v)


def _is_unit_free(x: object) -> bool:
    """Whether `x` is concrete and every element is zero or infinite."""
    if isinstance(x, jax.core.Tracer):
        return False
    a = np.asarray(x)
    return bool(np.all((a == 0) | np.isinf(a)))


def _check_bare(
    name: str, fn: Any, carrying: list[tuple[str, Any, int]], *, in_body: bool
) -> None:
    """Raise if a parameter that carries the site dimension is bare, not 0 or ±inf.

    `in_body` says the site is in a scan or cond body, where a bare tracer is
    often a carry that started bare.
    """
    has_q = any(_is_q(v) for _, v, _ in carrying)
    for path, v, _ in carrying:
        if _is_q(v) or _is_unit_free(v):
            continue
        k = path.rsplit(".", 1)[-1]
        if has_q:
            err = (
                f"Site '{name}': parameter '{k}' is bare next to Quantity "
                "parameters; give it units."
            )
        else:
            err = (
                f"Site '{name}': parameter '{k}' is bare, and "
                f"{type(unwrap(fn)).__name__} has a unit rule; give its "
                "parameters units instead of using units=."
            )
        # Inside a body every bare value is a tracer, even a concrete 0.0
        # start value, so the concrete-zero hint would not apply there.
        if isinstance(v, jax.core.Tracer) and in_body:
            err += (
                " Inside a scan or cond body, a value passed in bare stays "
                "bare: pass the scan's start value as a Quantity in the "
                "usys unit, or the cond's operand as a Quantity, or pass "
                "rewrap=False and add the units in the body."
            )
        elif isinstance(v, jax.core.Tracer):
            err += (
                " If it is a zero or an infinity, write it as a concrete 0.0 "
                "or np.zeros(...), not jnp.zeros(...) inside the model."
            )
        raise UnitfulError(err)


def _restore(old: Any, new: Any) -> None:
    """Copy onto `new` the state that NumPyro's pytree round trip drops from `old`.

    Unflattening sets ``_validate_args`` from the instance dict, so it is None
    when validation comes from the class default, and log_prob then skips the
    support check. It also drops attributes that ``__init__`` sets outside the
    pytree fields, such as ``Independent.reparametrized_params``. Cached lazy
    properties stay dropped, since they hold the unstripped values.
    """
    if isinstance(old, dist.Distribution):
        new._validate_args = old._validate_args  # noqa: SLF001
        for k, v in vars(old).items():
            cached = isinstance(getattr(type(old), k, None), lazy_property)
            if k not in vars(new) and not cached and not _has_quantity(v):
                setattr(new, k, v)

    def parts(t: Any) -> list[Any]:  # the leaves, stopping at sub-distributions
        def leaf(x: Any) -> bool:
            return (x is not t and isinstance(x, dist.Distribution)) or _is_q(x)

        return jax.tree.leaves(t, is_leaf=leaf)

    for a, b in zip(parts(old), parts(new), strict=True):
        if isinstance(a, dist.Distribution):
            _restore(a, b)


def _rebuild(fn: Any, f: Callable[[tuple[Any, ...], Any], Any]) -> Any:
    """Map `f` over the leaves of `fn`, keeping the state the rebuild drops."""
    new = jax.tree_util.tree_map_with_path(f, fn, is_leaf=_is_q)
    _restore(fn, new)
    return new


def _strip_params(name: str, fn: Any, unit: Any) -> Any:
    """Strip every Quantity in `fn`; return `fn` itself if it holds none.

    A parameter that carries the dimension is stripped in `unit` to its power.
    Any other Quantity is a shape parameter and must be dimensionless. Which is
    which goes by the leaf's path, since NumPyro may reshape a parameter into a
    new object while its support keeps the original.
    """
    if not _has_quantity(fn):
        return fn
    powers = strip_powers(fn) if unit is not None else {}

    def strip(kp: tuple[Any, ...], v: Any) -> Any:
        if not _is_q(v):
            return v
        path = leaf_path(fn, kp)
        if path in powers:
            return u.ustrip(unit ** powers[path], v)
        if u.dimension_of(v) != u.dimension("dimensionless"):
            err = (
                f"Site '{name}': shape parameter '{path.rsplit('.', 1)[-1]}' has "
                f"dimension {u.dimension_of(v)}; it must be dimensionless."
            )
            raise UnitfulError(err)
        return u.ustrip("", v)

    return _rebuild(fn, strip)


def _strip_scale(msg: dict[str, Any]) -> None:
    """Strip a dimensionless Quantity scale, as `handlers.scale` sets; raise if not."""
    name = msg["name"]
    if _is_q(msg.get("scale")):
        msg["scale"] = _dimensionless(name, "the scale", msg["scale"])
    pts = msg.get("plate_to_scale") or {}
    for k, sc in pts.items():
        if _is_q(sc):
            pts[k] = _dimensionless(name, "the scale", sc)


class _OuterStripper(Messenger):
    """Strip a Quantity value that a handler outside `unitful` set.

    Sits below every other handler on the stack, and just above every
    `handlers.block`, so it processes after the handlers that may set a value
    and post-processes before them: it strips the value before an outer
    `trace` stores it.
    """

    def __init__(self, owner: "unitful") -> None:
        self.owner = owner
        super().__init__(None)

    def process_message(self, msg: dict[str, Any]) -> None:
        if msg["type"] not in ("sample", "deterministic", "param"):
            return
        v, name = msg["value"], msg["name"]
        if msg["type"] == "sample":
            _strip_scale(msg)
        _check_container(name, v)
        if not _is_q(v):
            return
        # By message, not name: an outer `scope` may have renamed the site.
        unit = self.owner._msg_units.get(id(msg))  # noqa: SLF001
        if unit is None and u.dimension_of(v) == u.dimension("dimensionless"):
            msg["value"] = u.ustrip("", v)
            return
        if unit is None:
            err = f"Site '{name}': a Quantity value was set at a site with no unit."
            if msg["type"] == "param":
                err += (
                    " If a callable init returns it, return a bare value in the "
                    "unit of usys, or pass the Quantity as the init value."
                )
            raise UnitfulError(err)
        if u.dimension_of(v) != u.dimension_of(unit):
            err = (
                f"Site '{name}': value has dimension {u.dimension_of(v)}, "
                f"expected {u.dimension_of(unit)}."
            )
            raise UnitfulError(err)
        msg["value"] = u.ustrip(unit, v)

    # The post-process pass checks values set while the site ran. A callable
    # `param` init that returns a Quantity is one: `unitful` saw only the
    # callable, recorded no unit, and this raises instead of letting the
    # Quantity reach the trace.
    postprocess_message = process_message


class unitful(Messenger):  # noqa: N801  (NumPyro handler naming)
    """Run a model whose distributions take Quantity parameters.

    At each sample site the parameters and any observed value are stripped in
    `usys`, so NumPyro computes on bare numbers and log p is per `usys` unit.
    The sampled value goes back to the model as a Quantity in the `usys` unit of
    the site's dimension (bare when ``rewrap=False``). ``units={"site": unit}``
    sets the dimension of a site whose family has no rule; the site is still
    expressed in `usys`. A ``units=`` value is a dimension (``"length"``) or
    the `usys` unit of one, because bare parameters are read in `usys` units.
    A key that names no sample site of the model raises `UnitfulError` when the
    model returns. VonMises sites are always in radians. The body of a
    `numpyro.contrib.control_flow.scan` and the branches of a
    `numpyro.contrib.control_flow.cond` run under a `unitful` of their own with
    the same `usys` and `rewrap`, so their sites are handled the same way.
    A custom SVI guide needs a `unitful` with the same `usys`, as in
    ``unitful(guide, model_handler.usys)``; a guide value replayed from another
    `usys`, or from a site in another unit, raises `UnitfulError`. `usys` may
    also be the name of a unit system, such as ``"galactic"``. A latent whose
    parameters are all `unxt.Angle` comes back as an `unxt.Angle`.

    Examples
    --------
    >>> import numpyro, numpyro.distributions as dist
    >>> from numpyro import handlers
    >>> import unxt as u
    >>> from unxts.interop.numpyro import unitful

    >>> def model():
    ...     return numpyro.sample("m", dist.Normal(u.Q(0.0, "pc"), u.Q(3.0, "pc")))

    >>> m = handlers.seed(unitful(model, u.unitsystems.galactic), 0)()
    >>> m.unit
    Unit("kpc")

    """

    def __init__(
        self,
        fn: Callable[..., Any] | None,
        usys: u.AbstractUnitSystem | str,
        *,
        rewrap: bool = True,
        units: Mapping[str, Any] | None = None,
    ) -> None:
        if fn is None:
            msg = "unitful needs a model: wrap it as unitful(model, usys)."
            raise TypeError(msg)
        if isinstance(usys, str):
            usys = u.unitsystem(usys)
        if not isinstance(usys, u.AbstractUnitSystem):
            msg = (
                "unitful needs a unit system, such as u.unitsystems.galactic "
                f"or 'galactic', got {usys!r}."
            )
            raise TypeError(msg)
        self.usys = usys
        self.rewrap = rewrap
        self._overrides = {
            k: self._override_dimension(k, v) for k, v in (units or {}).items()
        }
        self._site_units: dict[str, Any] = {}  # static units only
        # Sample-site names as `unitful` sees them, before any outer rename.
        self._seen: set[str] = set()
        # Unit of each message between process and postprocess, keyed by
        # id(msg): outer handlers such as `scope` may rename the site.
        self._msg_units: dict[int, Any] = {}
        # The tagged `infer` dict of each sample message, to see if `replay` swaps it.
        self._msg_infer: dict[int, Any] = {}
        # The Quantity type a sample site gives back, such as `u.Angle`.
        self._msg_cls: dict[int, type] = {}
        self._calls = 0  # depth of calls through `__call__`, not `with unitful(...)`
        self._strippers: list[list[_OuterStripper]] = []  # per active entry
        self._ran = False  # whether any message reached the handler
        self._in_body = False  # whether this runs a scan or cond body
        # The mark this handler leaves on a message. Bodies take their parent's,
        # since `cond` needs both branches to leave the same trace metadata.
        self._token = id(self)
        super().__init__(fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the model under this handler."""
        self._calls += 1
        try:
            return super().__call__(*args, **kwargs)
        finally:
            self._calls -= 1

    def __enter__(self) -> None:
        stack = prim._PYRO_STACK  # noqa: SLF001
        # A stripper at the bottom of the stack, and one just above every
        # `handlers.block`, since a block stops a message before it reaches the
        # handlers below it. Stripping is idempotent. NumPyro processes from
        # the top of the stack down and post-processes from the bottom up, so
        # each stripper goes above the strippers of enclosing `unitful`s: the
        # innermost one processes first and post-processes last. A scan or cond
        # body runs under its own block, so its only stripper goes just above
        # the innermost block, where the body's handlers start.
        blocks = [i + 1 for i, h in enumerate(stack) if isinstance(h, handlers.block)]
        floors = blocks[-1:] if self._in_body else [0, *blocks]
        new = []
        for floor in reversed(floors):  # top first, so lower indices hold
            i = next(
                (
                    i
                    for i, h in enumerate(stack)
                    if i >= floor and not isinstance(h, _OuterStripper)
                ),
                len(stack),
            )
            s = _OuterStripper(self)
            stack.insert(i, s)
            new.append(s)
        self._strippers.append(new)
        super().__enter__()

    def __exit__(self, *exc: object) -> None:
        new = self._strippers.pop()
        if not self._strippers:
            self._msg_units.clear()  # entries left by a site that raised
            self._msg_infer.clear()
            self._msg_cls.clear()
        try:
            super().__exit__(*exc)
        finally:
            stack = prim._PYRO_STACK  # noqa: SLF001
            stack[:] = [h for h in stack if not any(h is s for s in new)]
        missing = sorted(self._overrides.keys() - self._seen)
        if exc[0] is None and missing and not self._in_body:
            renamed = [k for k in missing if k in self._site_units]
            if renamed:
                msg = (
                    f"units= names {renamed}, which are site names after a handler "
                    "outside unitful renamed them. Key units= by the name inside "
                    "unitful, or put the scope inside unitful."
                )
            else:
                msg = (
                    f"units= names {missing}, which no call of the model has "
                    "reached yet."
                )
            raise UnitfulError(msg)

    @property
    def site_units(self) -> Mapping[str, Any]:
        """Read-only map of site name to unit; None marks a unitless site."""
        return MappingProxyType(self._site_units)

    def _override_dimension(self, name: str, v: Any) -> Any:
        """Return the dimension a ``units=`` value gives; a unit must be in `usys`."""
        if isinstance(v, u.AbstractDimension):
            return v
        if isinstance(v, str):
            try:
                return u.dimension(v)
            except ValueError:
                pass
        try:
            unit = u.unit(v)
        except (NotFoundLookupError, TypeError, ValueError) as e:
            msg = (
                f"Site '{name}': units= takes a dimension or a unit, such as "
                f"'length' or 'kpc', got {v!r}."
            )
            raise UnitfulError(msg) from e
        dim = u.dimension_of(unit)
        want = self._unit(dim, name)
        if unit != want:
            msg = (
                f"Site '{name}': units= gives {unit}, but bare parameters are read "
                f"in the unit of usys; pass units={{'{name}': '{want}'}}."
            )
            raise UnitfulError(msg)
        return dim

    def _dimension(self, fn: Any, name: str) -> Any:
        if name not in self._overrides:
            return site_dimension(fn, name, self.usys)
        dim = self._overrides[name]
        if not carrying_params(fn) and _dimensioned(fn):
            msg = (
                f"Site '{name}': units= needs bare parameters, but the "
                "distribution has Quantity parameters with dimensions and no "
                "unit rule."
            )
            raise UnitfulError(msg)
        rule = site_dimension(fn, name, self.usys) if carrying_params(fn) else None
        if rule is not None and rule != dim:
            msg = f"Site '{name}': units= gives {dim}, the distribution gives {rule}."
            raise UnitfulError(msg)
        return dim

    def _unit(self, dim: Any, name: str) -> Any:
        if dim is None:
            return None
        try:
            return self.usys[dim]
        except (KeyError, ValueError) as e:
            msg = f"Site '{name}': the unit system has no unit for {dim}."
            raise UnitfulError(msg) from e

    def _record(self, name: str, unit: Any) -> None:
        if self._site_units.setdefault(name, unit) != unit:
            prev = self._site_units[name]
            err = (
                f"Site '{name}': has unit {prev} in one place and {unit} in "
                "another. A site needs one unit in every call and in both "
                "branches of a cond."
            )
            raise UnitfulError(err)

    def _mark(self, msg: dict[str, Any], unit: Any) -> None:
        """Record the unit of `msg` by site name and by message."""
        self._record(msg["name"], unit)
        self._msg_units[id(msg)] = unit

    def _strip_value(self, msg: dict[str, Any], v: Any) -> Any:
        """Strip a Quantity in `usys`, recording the site's unit; bare passes."""
        _check_container(msg["name"], v)
        if not _is_q(v):
            self._mark(msg, None)
            return v
        unit = self._unit(u.dimension_of(v), msg["name"])
        self._mark(msg, unit)
        return u.ustrip(self.usys, v)

    def _no_override(self, name: str, kind: str) -> None:
        if name in self._overrides:
            msg = f"Site '{name}': units= applies to sample sites, not {kind} sites."
            raise UnitfulError(msg)

    def _process_factor(self, msg: dict[str, Any]) -> None:
        # `plate` and `mask` wrap the `Unit` distribution, so look underneath.
        self._no_override(msg["name"], "factor")
        name = msg["name"]
        lf = unwrap(msg["fn"]).log_factor
        if not _is_q(lf):
            return
        _dimensionless(name, "log_factor", lf)
        msg["fn"] = _rebuild(
            msg["fn"],
            lambda _, v: _dimensionless(name, "log_factor", v) if _is_q(v) else v,
        )

    def process_message(self, msg: dict[str, Any]) -> None:
        """Strip the units of a site and record the site's unit."""
        self._ran = True
        if msg["type"] == "control_flow":
            self._process_control_flow(msg)
            return
        if msg.get("_control_flow_done", False) and "_unitful" in msg:
            return  # a scan or cond site that a body handler already stripped
        if msg["type"] in ("sample", "deterministic", "param"):
            # dict.copy keeps the mark, so a handler outside that re-sends a
            # copy of a processed message is caught here.
            if self._token in msg.get("_unitful", ()):
                err = (
                    f"Site '{msg['name']}': this unitful already stripped the "
                    "site. Either the model calls the same unitful instance "
                    "again (make a new unitful, or call the bare model), or a "
                    "handler outside sent the site again, as handlers.do does "
                    "(put it inside: unitful(handlers.do(model, ...), usys))."
                )
                raise UnitfulError(err)
            msg["_unitful"] = (*msg.get("_unitful", ()), self._token)
        if msg["type"] == "deterministic":
            self._no_override(msg["name"], "deterministic")
            msg["value"] = self._strip_value(msg, msg["value"])
            return
        if msg["type"] == "param":
            self._process_param(msg)
            return
        if msg["type"] != "sample":
            return
        _strip_scale(msg)  # from handlers.scale inside unitful
        if isinstance(unwrap(msg["fn"]), dist.Unit):
            self._process_factor(msg)
        else:
            self._process_sample(msg)

    def _process_param(self, msg: dict[str, Any]) -> None:
        """Strip a param's init value, and its constraint's bounds in the same unit."""
        name = msg["name"]
        self._no_override(name, "param")
        if msg["args"]:
            msg["args"] = (self._strip_value(msg, msg["args"][0]),)
        else:
            self._mark(msg, None)  # a callable init
        con = msg["kwargs"].get("constraint")
        if not any(_is_q(x) for x in jax.tree.leaves(con, is_leaf=_is_q)):
            return
        unit = self._msg_units[id(msg)]
        if unit is None:
            err = (
                f"Site '{name}': the constraint has Quantity bounds, but the init "
                "value is bare; pass the init value as a Quantity."
            )
            raise UnitfulError(err)

        def strip(v: Any) -> Any:
            if not _is_q(v):
                return v
            if u.dimension_of(v) != u.dimension_of(unit):
                err = (
                    f"Site '{name}': a constraint bound has dimension "
                    f"{u.dimension_of(v)}, expected {u.dimension_of(unit)}."
                )
                raise UnitfulError(err)
            return u.ustrip(unit, v)

        con = jax.tree.map(strip, con, is_leaf=_is_q)
        msg["kwargs"] = {**msg["kwargs"], "constraint": con}

    def _process_control_flow(self, msg: dict[str, Any]) -> None:
        """Run the body of a `scan` or the branches of a `cond` under a `unitful`.

        Both run their functions under `handlers.block`, so this handler never
        sees those sites while they run, and replay them afterwards with the
        values already set. A handler inside the block strips them as they run.
        """
        if msg["fn"] is scan_wrapper:
            msg["args"] = (self._body(msg["args"][0]), *msg["args"][1:])
        elif msg["fn"] is cond_wrapper:
            pred, t, f, *rest = msg["args"]
            msg["args"] = (pred, self._body(t), self._body(f), *rest)

    def _body(self, fn: Callable[..., Any]) -> "unitful":
        body = unitful(fn, self.usys, rewrap=self.rewrap)
        # Share the records, so units= and `site_units` cover the body's sites.
        body._overrides = self._overrides
        body._site_units = self._site_units
        body._seen = self._seen
        body._in_body = True
        body._token = self._token
        return body

    def _process_sample(self, msg: dict[str, Any]) -> None:
        name, fn = msg["name"], msg["fn"]
        self._seen.add(name)
        dim = self._dimension(fn, name)
        # VonMises works in radians whatever the usys angle unit is.
        if dim is not None and isinstance(unwrap(fn), dist.VonMises):
            unit = u.unit("rad")
        else:
            unit = self._unit(dim, name)
        # Tag the site, so a guide in another usys or unit is caught when
        # replayed here. The innermost unitful's tag stays: nested ones keep
        # the same dict.
        infer = msg.get("infer", {})
        if _TAG_KEY not in infer:
            msg["infer"] = {**infer, _TAG_KEY: (self.usys, unit)}
        self._msg_infer[id(msg)] = msg["infer"]

        # A dimensionless site is unambiguous, so bare numbers there are fine.
        none = u.dimension("dimensionless")
        checked = dim is not None and dim != none
        _check_bare(
            name, fn, carrying_params(fn) if checked else [], in_body=self._in_body
        )

        # Any value here comes from obs= or an inner handler: inference handlers
        # sit outside `unitful` and set the value after this runs.
        value = msg["value"]
        _check_container(name, value)
        if _is_q(value):
            expected = dim if dim is not None else none
            if u.dimension_of(value) != expected:
                err = (
                    f"Site '{name}': value has dimension {u.dimension_of(value)}, "
                    f"expected {expected}."
                )
                raise UnitfulError(err)
        elif checked and value is not None:
            # VonMises is read in radians whatever the usys angle unit is.
            vm = isinstance(unwrap(fn), dist.VonMises)
            of_usys = "" if vm else ", the unit of usys"
            err = (
                f"Site '{name}': value is bare. A bare value would be read in "
                f"{unit}{of_usys}; wrap the data in its own unit, as in "
                "u.Q(value, <your unit>)."
            )
            if not self._calls:
                err += (
                    " unitful is in use as a context manager, so handlers such as "
                    "log_density's substitute run inside it and set bare values. "
                    "Wrap the model instead: log_density(unitful(model, usys), ...)."
                )
            raise UnitfulError(err)

        self._mark(msg, unit)
        qs = [v for _, v, _ in carrying_params(fn) if _is_q(v)]
        if qs and all(isinstance(v, u.Angle) for v in qs):
            self._msg_cls[id(msg)] = u.Angle

        msg["fn"] = _strip_params(name, fn, unit)
        if _is_q(value):
            msg["value"] = u.ustrip("" if unit is None else unit, value)

    def _check_replayed(self, msg: dict[str, Any], unit: Any) -> None:
        """Raise if `replay` brought this site a value made in another usys or unit.

        A bare site, in the guide or here, has no unit to compare.
        """
        infer = self._msg_infer.pop(id(msg), None)
        new = msg.get("infer", {})  # `reparam` drops the key
        if infer is None or new is infer or _TAG_KEY not in new:
            return
        other, other_unit = new[_TAG_KEY]
        if other != self.usys:
            err = (
                f"Site '{msg['name']}': the replayed value (for example from an "
                f"SVI guide) was made under unitful with usys {other}, but the "
                f"model runs under {self.usys}. Bare values are read in the usys "
                "unit, so both must share one usys: unitful(guide, "
                "model_handler.usys)."
            )
            raise UnitfulError(err)
        if None not in (unit, other_unit) and other_unit != unit:
            err = (
                f"Site '{msg['name']}': the replayed value (for example from an "
                f"SVI guide) is in {other_unit}, but the model reads this site "
                f"in {unit}. "
            )
            if u.dimension_of(unit) != u.dimension_of(other_unit):
                err += f"Give the guide's site the dimension {u.dimension_of(unit)}."
            else:
                err += (
                    "A VonMises site is always in rad, so the guide and the "
                    "model need a VonMises at this site in both, or a usys "
                    "whose angle unit is rad."
                )
            raise UnitfulError(err)

    def postprocess_message(self, msg: dict[str, Any]) -> None:
        """Give a site's value back to the model as a Quantity."""
        if id(msg) not in self._msg_units:
            return
        unit = self._msg_units.pop(id(msg))
        cls = self._msg_cls.pop(id(msg), u.Q)
        self._check_replayed(msg, unit)
        # An outer `scope` may have renamed the site since `process_message`.
        self._record(msg["name"], unit)
        if _is_q(msg["value"]):
            err = (
                f"Site '{msg['name']}': a Quantity value set outside unitful "
                "reached the site unstripped. Put the handler that set it "
                "inside unitful, as unitful(handler(model, ...), usys)."
            )
            raise UnitfulError(err)
        if unit is not None and self.rewrap:
            msg["value"] = cls(msg["value"], unit)
