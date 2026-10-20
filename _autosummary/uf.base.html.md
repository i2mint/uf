# uf.base

Core functionality for uf - UI Fast.

This module provides the main `mk_rjsf_app` function that bridges:

- Functions → HTTP services (via qh)
- HTTP services → Web UI forms (via ju.rjsf)

Following the “convention over configuration” philosophy, it provides sensible
defaults while allowing customization where needed.

### Functions

| [`mk_rjsf_app`](#uf.base.mk_rjsf_app)(funcs, \*[, config, input_trans, ...])   | Create an RJSF-backed web app from a list of functions.   |
|-------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|

### Classes

| [`UfApp`](#uf.base.UfApp)(funcs, \*\*mk_rjsf_app_kwargs)   | Wrapper class for uf applications.   |
|-----------------------------------------------------------------------------------------|--------------------------------------|

### *class* uf.base.UfApp(funcs, \*\*mk_rjsf_app_kwargs)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Wrapper class for uf applications.

Provides a higher-level interface with additional conveniences
beyond the raw qh app.

#### app

The underlying qh/bottle/fastapi app

#### function_specs

FunctionSpecStore for function metadata

#### funcs

Dictionary mapping function names to callables

#### call(func_name, \*\*kwargs)

Call a function directly by name.

* **Parameters:**
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the function to call
  * **\*\*kwargs** – Arguments to pass to the function
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
* **Returns:**
  Result of the function call
* **Raises:**
  [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – If function name not found

#### get_spec(func_name)

Get RJSF specification for a function.

* **Parameters:**
  **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the function
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary with schema and uiSchema
* **Raises:**
  [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – If function name not found

#### list_functions()

Get list of all function names.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  List of function name strings

#### run(host='localhost', port=8080, \*\*kwargs)

Run the web application.

* **Parameters:**
  * **host** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Host to bind to
  * **port** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Port to listen on
  * **\*\*kwargs** – Additional arguments passed to app.run()

### uf.base.mk_rjsf_app(funcs, , config=None, input_trans=None, output_trans=None, rjsf_config=None, ui_schema_factory=None, param_to_prop_type=None, page_title='Function Interface', function_display_names=None, custom_css=None, rjsf_theme='default', add_ui=True, \*\*qh_kwargs)

Create an RJSF-backed web app from a list of functions.

This is the main entry point for uf. It combines qh’s HTTP service
generation with ju’s RJSF form generation to create a complete
web interface for your functions.

* **Parameters:**
  * **funcs** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)], [`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]]) – Iterable of callables to expose via web UI – functions, partials,
    callable instances (named by `uf.specs.name_of_func`) – or a mapping
    `{name: callable}` to name them yourself. The name is the route and the
    key of the form spec
  * **config** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Optional qh.AppConfig for HTTP service configuration
  * **input_trans** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Optional input transformation function for qh
  * **output_trans** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Optional output transformation function for qh
  * **rjsf_config** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Optional configuration dict for RJSF generation
  * **ui_schema_factory** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Optional callable to customize UI schema
  * **param_to_prop_type** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Optional callable to map parameters to types
  * **page_title** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Title for the web interface
  * **function_display_names** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)]) – Optional mapping to override function names
  * **custom_css** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional custom CSS for the web interface
  * **rjsf_theme** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – RJSF theme to use (‘default’, ‘material-ui’, etc.)
  * **add_ui** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to add UI routes (set False for API-only)
  * **\*\*qh_kwargs** – Additional keyword arguments passed to qh.mk_app
* **Returns:**
  - HTTP endpoints for each function
  - RJSF-based web interface (if add_ui=True)
  - API routes for function specs
* **Return type:**
  A configured web application (bottle or FastAPI app) with

### Example

```pycon
>>> def add(x: int, y: int) -> int:
...     '''Add two numbers'''
...     return x + y
...
>>> def greet(name: str) -> str:
...     '''Greet a person'''
...     return f"Hello, {name}!"
...
>>> app = mk_rjsf_app([add, greet])
>>> # app.run()  # Start the web server
```

Example with customization:

```pycon
>>> from uf import mk_rjsf_app, RjsfFieldConfig
>>>
>>> def send_email(to: str, subject: str, body: str):
...     '''Send an email'''
...     pass
...
>>> app = mk_rjsf_app(
...     [send_email],
...     page_title="Email Sender",
...     custom_css="body { background: #f0f0f0; }",
... )
```
