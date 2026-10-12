# uf.trans

Input transformation integration for uf.

Provides a registry for custom type transformations that bridges between
RJSF form data and qh’s input transformation system.

### Functions

| [`get_global_registry`](#uf.trans.get_global_registry)()             | Get the global transformation registry.   |
|------------------------------------------------------------------------------------|-------------------------------------------|
| [`register_common_types`](#uf.trans.register_common_types)()           | Register commonly-used Python types.      |
| [`register_type`](#uf.trans.register_type)(\*args, \*\*kwargs) | Register a type in the global registry.   |

### Classes

| [`InputTransformRegistry`](#uf.trans.InputTransformRegistry)()   | Registry for custom type transformations.   |
|-----------------------------------------------------------------------------|---------------------------------------------|

### *class* uf.trans.InputTransformRegistry

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Registry for custom type transformations.

Integrates with qh’s type registry and extends it for UI needs,
allowing custom types to be properly handled in both the form
interface and the HTTP service layer.

### Example

```pycon
>>> from datetime import datetime
>>> registry = InputTransformRegistry()
>>>
>>> # Register a custom type
>>> registry.register_type(
...     datetime,
...     to_json=lambda dt: dt.isoformat(),
...     from_json=lambda s: datetime.fromisoformat(s),
...     ui_widget='datetime'
... )
```

#### from_json(value, py_type)

Transform a JSON value to Python type.

* **Parameters:**
  * **value** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – JSON value to transform
  * **py_type** ([`Type`](https://docs.python.org/3/library/typing.html#typing.Type)) – Target Python type
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
* **Returns:**
  Transformed value

#### get_all_registered_types()

Get list of all registered types.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Type`](https://docs.python.org/3/library/typing.html#typing.Type)]
* **Returns:**
  List of registered Python types

#### get_handler(py_type)

Get handler for a type.

* **Parameters:**
  **py_type** ([`Type`](https://docs.python.org/3/library/typing.html#typing.Type)) – Python type to look up
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]
* **Returns:**
  Handler dict or None if not registered

#### get_ui_config(py_type)

Get UI configuration for a type.

* **Parameters:**
  **py_type** ([`Type`](https://docs.python.org/3/library/typing.html#typing.Type)) – Python type to look up
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`RjsfFieldConfig`](uf.rjsf_config.md#uf.rjsf_config.RjsfFieldConfig)]
* **Returns:**
  RjsfFieldConfig or None if not registered

#### mk_input_trans_for_funcs(funcs)

Create input transformation compatible with qh.

This creates a transformation function that can be passed to
qh.mk_app as the input_trans parameter.

* **Parameters:**
  **funcs** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – List of functions to create transformation for
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)
* **Returns:**
  Transformation function for qh

### Example

```pycon
>>> from uf import mk_rjsf_app
>>> registry = InputTransformRegistry()
>>> # ... register types ...
>>> input_trans = registry.mk_input_trans_for_funcs([my_func])
>>> app = mk_rjsf_app([my_func], input_trans=input_trans)
```

#### mk_output_trans()

Create output transformation for qh.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)
* **Returns:**
  Transformation function for qh output

### Example

```pycon
>>> output_trans = registry.mk_output_trans()
>>> app = mk_rjsf_app([my_func], output_trans=output_trans)
```

#### register_type(py_type, , to_json=None, from_json=None, ui_widget=None, ui_config=None, json_schema_type=None, json_schema_format=None)

Register a type with both qh and UI configuration.

* **Parameters:**
  * **py_type** ([`Type`](https://docs.python.org/3/library/typing.html#typing.Type)) – Python type to register
  * **to_json** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Function to convert Python type to JSON-serializable
  * **from_json** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Function to convert JSON to Python type
  * **ui_widget** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – RJSF widget to use for this type
  * **ui_config** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`RjsfFieldConfig`](uf.rjsf_config.md#uf.rjsf_config.RjsfFieldConfig)]) – Full RjsfFieldConfig for this type
  * **json_schema_type** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – JSON Schema type (e.g., ‘string’, ‘number’)
  * **json_schema_format** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – JSON Schema format (e.g., ‘date-time’, ‘email’)
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### Example

```pycon
>>> from pathlib import Path
>>> registry.register_type(
...     Path,
...     to_json=str,
...     from_json=Path,
...     ui_widget='text',
...     json_schema_type='string'
... )
```

#### to_json(value, py_type=None)

Transform a Python value to JSON-serializable form.

* **Parameters:**
  * **value** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Value to transform
  * **py_type** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Type`](https://docs.python.org/3/library/typing.html#typing.Type)]) – Optional type hint (uses type(value) if not provided)
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
* **Returns:**
  JSON-serializable value

### uf.trans.get_global_registry()

Get the global transformation registry.

* **Return type:**
  [`InputTransformRegistry`](#uf.trans.InputTransformRegistry)
* **Returns:**
  The global InputTransformRegistry instance

### uf.trans.register_common_types()

Register commonly-used Python types.

This includes:

- datetime.datetime
- datetime.date
- datetime.time
- pathlib.Path
- uuid.UUID
- decimal.Decimal

### uf.trans.register_type(\*args, \*\*kwargs)

Register a type in the global registry.

This is a convenience function that uses the global registry.
See InputTransformRegistry.register_type for full documentation.
