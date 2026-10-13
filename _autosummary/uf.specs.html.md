# uf.specs

Function specification management for uf.

Provides a Mapping-based interface to function specifications, including RJSF
form specs and OpenAPI schemas.

### Functions

| [`name_of_func`](#uf.specs.name_of_func)(func)   | The name a function is served under: its `__name__`; failing that, for a `functools.partial`, that of the function it wraps; for a callable instance, its class name.   |
|-----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### Classes

| [`FunctionSpecStore`](#uf.specs.FunctionSpecStore)(funcs, \*[, rjsf_config, ...])   | A mapping from function names to their RJSF specifications.   |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------|

### Exceptions

| [`JuNotInstalledWarning`](#uf.specs.JuNotInstalledWarning)         | Emitted when `ju` is absent and uf falls back to basic form specs.                             |
|--------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| [`PositionalOnlyParamWarning`](#uf.specs.PositionalOnlyParamWarning)    | A required positional-only parameter can't be filled by a keyword-only caller.                 |
| [`UnintrospectableCallableError`](#uf.specs.UnintrospectableCallableError) | A callable has no signature to build a form from (e.g. `max`, `getattr`).                      |
| [`UnservableCallableError`](#uf.specs.UnservableCallableError)       | A callable can never be called from a uf form (e.g. only required positional-only parameters). |

### *class* uf.specs.FunctionSpecStore(funcs, , rjsf_config=None, ui_schema_factory=None, param_to_prop_type=None, validate=False)

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

A mapping from function names to their RJSF specifications.

Lazily generates and caches form specs for each function.

* **Parameters:**
  * **funcs** (`Union`[[`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)], [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]]) – Iterable of callables to generate specs for (named by `name_of_func`:
    `__name__`, the wrapped function’s for a `functools.partial`, the class
    name for a callable instance), or a mapping `{name: callable}` to name
    them explicitly
  * **rjsf_config** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Optional configuration dict for RJSF generation
  * **ui_schema_factory** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Optional callable to customize UI schema generation
  * **param_to_prop_type** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Optional callable to map parameters to property types
  * **validate** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If true, every function is checked at construction, so a callable
    uf can’t build a form for fails there, naming it, rather than on the first
    request (default false: checked on first access of its spec;
    `mk_rjsf_app` validates).

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

### *exception* uf.specs.UnservableCallableError

Bases: [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError)

A callable can never be called from a uf form (e.g. only required
positional-only parameters).

### uf.specs.name_of_func(func)

The name a function is served under: its `__name__`; failing that, for a
`functools.partial`, that of the function it wraps; for a callable instance, its
class name.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> import functools
>>> name_of_func(len)
'len'
>>> name_of_func(functools.partial(int, base=2))
'int'
>>> class Doubler:
...     def __call__(self, x): return 2 * x
>>> name_of_func(Doubler())
'Doubler'
```
