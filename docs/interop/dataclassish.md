# 🧱 Dataclassish

This guide demonstrates how `unxt` types work with the `dataclassish` library for introspection and manipulation of unitful quantities and related types.

`unxt` types (dimensions, units, unit systems, and quantities) are all Equinox modules, which are Python dataclasses. This means they work seamlessly with `dataclassish` functions for field manipulation and introspection.

## Summary

`dataclassish` provides convenient tools for introspecting and manipulating `unxt` types:

- **`fields(obj)`**: Get all field information for a quantity or related type
- **`field_keys(obj)`**: Iterate over field names
- **`field_values(obj)`**: Iterate over field values
- **`field_items(obj)`**: Iterate over (name, value) pairs
- **`asdict(obj)`**: Convert to a dictionary representation
- **`astuple(obj)`**: Convert to a tuple of values
- **`get_field(obj, name)`**: Access a specific field by name
- **`replace(obj, \*\*kwargs)`**: Create a copy with modified fields

All of these work seamlessly across:

- Dimensions
- Units
- Unit Systems
- Quantity types (Quantity, ParametricQuantity, Angle, etc.)

This makes it easy to build generic code that works with unxt types without needing to know their internal structure.

## Setup

```pycon
>>> import unxt as u
>>> import dataclassish as dc

>>> print(f"unxt version: {u.__version__}")
unxt version: ...
>>> print(f"dataclassish version: {dc.__version__}")
dataclassish version: ...
```

## Dimensions

Let's start by exploring how `dataclassish` works with `unxt` dimensions.

```pycon
>>> # Create a dimension
>>> length_dim = u.dimension("length")
>>> print(f"Dimension: {length_dim}")
Dimension: length
>>> print(f"Type: {type(length_dim)}")
Type: <class 'astropy.units.physical.PhysicalType'>

>>> # Get fields
>>> fields = dc.fields(length_dim)
>>> [f.name for f in fields]
['_unit', '_physical_type']

>>> # Get field keys, values, and items
>>> print(f"Field keys: {list(dc.field_keys(length_dim))}")
Field keys: ['_unit', '_physical_type']
>>> print(f"Field values: {list(dc.field_values(length_dim))}")
Field values: [Unit("m"), ['length']]
>>> print(f"Field items: {list(dc.field_items(length_dim))}")
Field items: [('_unit', Unit("m")), ('_physical_type', ['length'])]
```

## Units

Now let's explore units with `dataclassish`.

```pycon
>>> # Create a unit
>>> meter = u.unit("m")
>>> print(f"Unit: {meter}")
Unit: m
>>> print(f"Type: {type(meter)}")
Type: <class 'astropy.units.core.IrreducibleUnit'>

>>> # Get fields
>>> fields = dc.fields(meter)
>>> [f.name for f in fields]
['_names']

>>> # Get field keys, values, items
>>> print(f"Field keys: {list(dc.field_keys(meter))}")
Field keys: ['_names']
>>> print(f"Field values: {list(dc.field_values(meter))}")
Field values: [['m', 'meter']]
>>> print(f"Field items: {list(dc.field_items(meter))}")
Field items: [('_names', ['m', 'meter'])]
```

## Unit Systems

Let's explore unit systems with `dataclassish`.

```pycon
>>> # Create a unit system
>>> si = u.unitsystem("si")
>>> print(f"Unit System: {si}")
Unit System: SIUnitSystem(length, mass, time, amount, electric_current, temperature, luminous_intensity, angle)
>>> print(f"Type: {type(si)}")
Type: <class 'unxt...unitsystems...SIUnitSystem'>

>>> # Get fields
>>> fields = dc.fields(si)
>>> [f.name for f in fields]
['length', 'mass', 'time', 'amount', 'electric_current', 'temperature', 'luminous_intensity', 'angle']

>>> # Get field keys, values, items
>>> print(f"Field keys: {list(dc.field_keys(si))}")
Field keys: ['length', 'mass', 'time', 'amount', 'electric_current', 'temperature', 'luminous_intensity', 'angle']
>>> print(f"Field values: {list(dc.field_values(si))}")
Field values: [Unit("m"), Unit("kg"), Unit("s"), Unit("mol"), Unit("A"), Unit("K"), Unit("cd"), Unit("rad")]
>>> print(f"Field items: {list(dc.field_items(si))}")
Field items: [('length', Unit("m")), ('mass', Unit("kg")), ('time', Unit("s")), ('amount', Unit("mol")), ('electric_current', Unit("A")), ('temperature', Unit("K")), ('luminous_intensity', Unit("cd")), ('angle', Unit("rad"))]
```

```pycon
>>> # Convert to dict and tuple
>>> ussystem_dict = dc.asdict(si)
>>> print(f"asdict(unit_system): {ussystem_dict}")
asdict(unit_system): {'length': Unit("m"), 'mass': Unit("kg"), 'time': Unit("s"), 'amount': Unit("mol"), 'electric_current': Unit("A"), 'temperature': Unit("K"), 'luminous_intensity': Unit("cd"), 'angle': Unit("rad")}

>>> ussystem_tuple = dc.astuple(si)
>>> print(f"astuple(unit_system): {ussystem_tuple}")
astuple(unit_system): (Unit("m"), Unit("kg"), Unit("s"), Unit("mol"), Unit("A"), Unit("K"), Unit("cd"), Unit("rad"))
```

## Quantities

Now let's explore how `dataclassish` works with quantities. We'll examine different quantity types.

### Basic Quantity

The default `Quantity` type is the lightweight class wrapping unit information with a value.

```pycon
>>> # Create a quantity
>>> distance = u.Q(10.0, "m")
>>> print(f"Quantity: {distance}")
Quantity: Quantity(10., unit='m')
>>> print(f"Type: {type(distance)}")
Type: <class 'unxt...quantity...Quantity'>

>>> # Get fields
>>> fields = dc.fields(distance)
>>> [f.name for f in fields]
['value', 'unit']

>>> # Get field keys, values, items
>>> print(f"Field keys: {list(dc.field_keys(distance))}")
Field keys: ['value', 'unit']
>>> print(f"Field values: {list(dc.field_values(distance))}")
Field values: [Array(10., dtype=float32...), Unit("m")]
>>> print(f"Field items: {list(dc.field_items(distance))}")
Field items: [('value', Array(10., dtype=float32...)), ('unit', Unit("m"))]
```

```pycon
>>> # Convert to dict and tuple
>>> qty_dict = dc.asdict(distance)
>>> print(f"asdict(quantity): {qty_dict}")
asdict(quantity): {'value': Array(10., dtype=float32...), 'unit': Unit("m")}

>>> qty_tuple = dc.astuple(distance)
>>> print(f"astuple(quantity): {qty_tuple}")
astuple(quantity): (Array(10., dtype=float32...), Unit("m"))

>>> # Use replace() to modify a quantity
>>> new_distance = dc.replace(distance, value=20.0)
>>> print(f"Original: {distance}")
Original: Quantity(10., unit='m')
>>> print(f"After replace(value=20.0): {new_distance}")
After replace(value=20.0): Quantity(20., unit='m')
```

```pycon
>>> # Use get_field() to access individual fields
>>> value_field = dc.get_field(distance, "value")
>>> print(f"get_field(distance, 'value'): {value_field}")
get_field(distance, 'value'): 10.0

>>> unit_field = dc.get_field(distance, "unit")
>>> print(f"get_field(distance, 'unit'): {unit_field}")
get_field(distance, 'unit'): m
```

### Angle (specialized quantity with wrapping)

`Angle` is a specialized quantity type for angular measurements with automatic wrapping.

```pycon
>>> # Create an Angle quantity
>>> theta = u.Angle(45.0, "deg")
>>> print(f"Angle: {theta}")
Angle: Angle(45., unit='deg')
>>> print(f"Type: {type(theta)}")
Type: <class 'unxt...quantity...Angle'>

>>> # Dataclassish introspection
>>> print(f"Fields: {[f.name for f in dc.fields(theta)]}")
Fields: ['value', 'unit']
>>> print(f"Field keys: {list(dc.field_keys(theta))}")
Field keys: ['value', 'unit']
>>> print(f"asdict: {dc.asdict(theta)}")
asdict: {'value': Array(45., dtype=float32...), 'unit': Unit("deg")}

>>> # Replace the angle value
>>> new_theta = dc.replace(theta, value=90.0)
>>> print(f"Original: {theta}")
Original: Angle(45., unit='deg')
>>> print(f"After replace(value=90.0): {new_theta}")
After replace(value=90.0): Angle(90., unit='deg')

>>> # Get individual fields
>>> angle_value = dc.get_field(theta, "value")
>>> angle_unit = dc.get_field(theta, "unit")
>>> print(f"Angle value: {angle_value}")
Angle value: 45.0
>>> print(f"Angle unit: {angle_unit}")
Angle unit: deg
```
