# uf.openapi

OpenAPI/Swagger integration for uf.

Automatically generates OpenAPI specifications and provides Swagger UI
for API documentation and testing.

### Functions

| [`add_openapi_routes`](#uf.openapi.add_openapi_routes)(app, funcs, \*\*spec_kwargs)   | Add OpenAPI routes to an app.                 |
|----------------------------------------------------------------------------------------------------|-----------------------------------------------|
| [`function_to_openapi_operation`](#uf.openapi.function_to_openapi_operation)(func[, path])       | Convert a function to OpenAPI operation spec. |
| [`generate_openapi_spec`](#uf.openapi.generate_openapi_spec)(funcs[, title, ...])        | Generate OpenAPI 3.0 specification.           |
| [`redoc_ui_html`](#uf.openapi.redoc_ui_html)([openapi_url])                      | Generate ReDoc UI HTML.                       |
| [`swagger_ui_html`](#uf.openapi.swagger_ui_html)([openapi_url])                    | Generate Swagger UI HTML.                     |

### Classes

| [`OpenAPIConfig`](#uf.openapi.OpenAPIConfig)([title, version, description, ...])   | Configuration for OpenAPI generation.   |
|------------------------------------------------------------------------------------------------------|-----------------------------------------|

### *class* uf.openapi.OpenAPIConfig(title='API', version='1.0.0', description='', servers=None, enable_swagger=True, enable_redoc=True)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Configuration for OpenAPI generation.

### Example

```pycon
>>> config = OpenAPIConfig(
...     title="My API",
...     version="2.0.0",
...     description="API for my application"
... )
```

#### to_dict()

Convert to dictionary.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Configuration dictionary

### uf.openapi.add_openapi_routes(app, funcs, \*\*spec_kwargs)

Add OpenAPI routes to an app.

Adds:

- /openapi.json: OpenAPI specification
- /docs: Swagger UI
- /redoc: ReDoc UI

* **Parameters:**
  * **app** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Web application
  * **funcs** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – List of functions
  * **\*\*spec_kwargs** – Arguments for generate_openapi_spec
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### uf.openapi.function_to_openapi_operation(func, path=None)

Convert a function to OpenAPI operation spec.

* **Parameters:**
  * **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to convert
  * **path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Optional API path
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  OpenAPI operation dictionary

### uf.openapi.generate_openapi_spec(funcs, title='API', version='1.0.0', description='', servers=None)

Generate OpenAPI 3.0 specification.

* **Parameters:**
  * **funcs** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – List of functions
  * **title** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – API title
  * **version** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – API version
  * **description** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – API description
  * **servers** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]) – Optional list of server configs
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  OpenAPI specification dictionary

### uf.openapi.redoc_ui_html(openapi_url='/openapi.json')

Generate ReDoc UI HTML.

* **Parameters:**
  **openapi_url** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – URL to OpenAPI spec
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  HTML string for ReDoc UI

### uf.openapi.swagger_ui_html(openapi_url='/openapi.json')

Generate Swagger UI HTML.

* **Parameters:**
  **openapi_url** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – URL to OpenAPI spec
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  HTML string for Swagger UI
