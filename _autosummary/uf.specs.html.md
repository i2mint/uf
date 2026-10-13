# uf.specs

Function specification management for uf.

Provides a Mapping-based interface to function specifications, including RJSF
form specs and OpenAPI schemas.

### Classes

| [`FunctionSpecStore`](#uf.specs.FunctionSpecStore)(funcs, \*[, rjsf_config, ...])   | A mapping from function names to their RJSF specifications.   |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------|

### Exceptions

| [`JuNotInstalledWarning`](#uf.specs.JuNotInstalledWarning)         | Emitted when `ju` is absent and uf falls back to basic form specs.             |
|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [`PositionalOnlyParamWarning`](#uf.specs.PositionalOnlyParamWarning)    | A required positional-only parameter can't be filled by a keyword-only caller. |
| [`UnintrospectableCallableError`](#uf.specs.UnintrospectableCallableError) | A callable has no signature to build a form from (e.g. `max`, `getattr`).      |

### *class* uf.specs.FunctionSpecStore(funcs, , rjsf_config=None, ui_schema_factory=None, param_to_prop_type=None)

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

A mapping from function names to their RJSF specifications.

Lazily generates and caches form specs for each function.

* **Parameters:**
  * **funcs** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Iterable of callable functions to generate specs for
  * **rjsf_config** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Optional configuration dict for RJSF generation
  * **ui_schema_factory** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Optional callable to customize UI schema generation
  * **param_to_prop_type** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Optional callable to map parameters to property types

### Example

```pycon
>>> def add(x: int, y: int) -> int:
...     '''Add two numbers'''
...     return x + y
>>> specs = FunctionSpecStore([add])
>>> 'add' in specs
True
>>> spec = specs['add']
>>> 'schema' in spec
True
```

#### *property* function_list *: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[dict](https://docs.python.org/3/builtins/stdtypes.html#dict)]*

Get list of all functions with basic metadata.

* **Returns:**
  List of dictionaries with function name and description

#### get_func(func_name)

Get the original function object by name.

* **Parameters:**
  **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the function
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)
* **Returns:**
  The function object
* **Raises:**
  [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – If function name not found

### *exception* uf.specs.JuNotInstalledWarning

Bases: [`UserWarning`](https://docs.python.org/3/builtins/exceptions.html#UserWarning)

Emitted when `ju` is absent and uf falls back to basic form specs.

### *exception* uf.specs.PositionalOnlyParamWarning

Bases: [`UserWarning`](https://docs.python.org/3/builtins/exceptions.html#UserWarning)

A required positional-only parameter can’t be filled by a keyword-only caller.

### *exception* uf.specs.UnintrospectableCallableError

Bases: [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError)

A callable has no signature to build a form from (e.g. `max`, `getattr`).
