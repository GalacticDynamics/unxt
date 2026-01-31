# Dataclassish Interoperability

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
- **`replace(obj, **kwargs)`**: Create a copy with modified fields

All of these work seamlessly across:

- Dimensions
- Units
- Unit Systems
- Quantity types (Quantity, BareQuantity, Angle, Distance, etc.)

This makes it easy to build generic code that works with unxt types without needing to know their internal structure.


## Setup

```python
import unxt as u
import dataclassish as dc

print(f"unxt version: {u.__version__}")
print(f"dataclassish version: {dc.__version__}")
```

## Dimensions

Let's start by exploring how `dataclassish` works with `unxt` dimensions.

```python
# Create a dimension
length_dim = u.dimension("length")
print(f"Dimension: {length_dim}")
print(f"Type: {type(length_dim)}")

# Get fields
fields = dc.fields(length_dim)
print(f"\nFields: {fields}")

# Get field keys, values, and items
print(f"\nField keys: {list(dc.field_keys(length_dim))}")
print(f"Field values: {list(dc.field_values(length_dim))}")
print(f"Field items: {list(dc.field_items(length_dim))}")
```

## Units

Now let's explore units with `dataclassish`.

```python
# Create a unit
meter = u.unit("m")
print(f"Unit: {meter}")
print(f"Type: {type(meter)}")

# Get fields
fields = dc.fields(meter)
print(f"\nFields: {fields}")

# Get field keys, values, items
print(f"\nField keys: {list(dc.field_keys(meter))}")
print(f"Field values: {list(dc.field_values(meter))}")
print(f"Field items: {list(dc.field_items(meter))}")
```


## Unit Systems

Let's explore unit systems with `dataclassish`.

```python
# Create a unit system
si = u.unitsystem("si")
print(f"Unit System: {si}")
print(f"Type: {type(si)}")

# Get fields
fields = dc.fields(si)
print(f"\nFields: {fields}")

# Get field keys, values, items
print(f"\nField keys: {list(dc.field_keys(si))}")
print(f"Field values: {list(dc.field_values(si))}")
print(f"Field items: {list(dc.field_items(si))}")
```

```python
# Convert to dict and tuple
ussystem_dict = dc.asdict(si)
print(f"asdict(unit_system): {ussystem_dict}")

ussystem_tuple = dc.astuple(si)
print(f"astuple(unit_system): {ussystem_tuple}")
```

## Quantities

Now let's explore how `dataclassish` works with quantities. We'll examine different quantity types.

### Basic Quantity (with dimension checking)

The `Quantity` type includes dimension parametrization and dimension checking.

```python
# Create a quantity with dimension checking
distance = u.Quantity(10.0, "m")
print(f"Quantity: {distance}")
print(f"Type: {type(distance)}")

# Get fields
fields = dc.fields(distance)
print(f"\nFields: {fields}")

# Get field keys, values, items
print(f"\nField keys: {list(dc.field_keys(distance))}")
print(f"Field values: {list(dc.field_values(distance))}")
print(f"Field items: {list(dc.field_items(distance))}")
```

```python
# Convert to dict and tuple
qty_dict = dc.asdict(distance)
print(f"asdict(quantity): {qty_dict}")

qty_tuple = dc.astuple(distance)
print(f"astuple(quantity): {qty_tuple}")

# Use replace() to modify a quantity
new_distance = dc.replace(distance, value=20.0)
print(f"\nOriginal: {distance}")
print(f"After replace(value=20.0): {new_distance}")
```

```python
# Use get_field() to access individual fields
value_field = dc.get_field(distance, "value")
print(f"get_field(distance, 'value'): {value_field}")

unit_field = dc.get_field(distance, "unit")
print(f"get_field(distance, 'unit'): {unit_field}")
```

### BareQuantity (lightweight, no dimension checking)

```python
# BareQuantity is a lightweight alternative without dimension checking
bare_qty = u.quantity.BareQuantity(5.0, "km")
print(f"BareQuantity: {bare_qty}")
print(f"Type: {type(bare_qty)}")

# All dataclassish functions work the same
print(f"\nFields: {dc.fields(bare_qty)}")
print(f"Field keys: {list(dc.field_keys(bare_qty))}")
print(f"Field values: {list(dc.field_values(bare_qty))}")
print(f"asdict: {dc.asdict(bare_qty)}")
print(f"astuple: {dc.astuple(bare_qty)}")

# Replace works here too
new_bare_qty = dc.replace(bare_qty, value=10.0)
print(f"\nOriginal: {bare_qty}")
print(f"After replace(value=10.0): {new_bare_qty}")
```

### Angle (specialized quantity with wrapping)

`Angle` is a specialized quantity type for angular measurements with automatic wrapping.

```python
# Create an Angle quantity
theta = u.Angle(45.0, "deg")
print(f"Angle: {theta}")
print(f"Type: {type(theta)}")

# Dataclassish introspection
print(f"\nFields: {dc.fields(theta)}")
print(f"Field keys: {list(dc.field_keys(theta))}")
print(f"asdict: {dc.asdict(theta)}")

# Replace the angle value
new_theta = dc.replace(theta, value=90.0)
print(f"\nOriginal: {theta}")
print(f"After replace(value=90.0): {new_theta}")

# Get individual fields
angle_value = dc.get_field(theta, "value")
angle_unit = dc.get_field(theta, "unit")
print(f"\nAngle value: {angle_value}")
print(f"Angle unit: {angle_unit}")
```
