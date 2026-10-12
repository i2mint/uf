# uf.routes

API routes for uf web interface.

Provides convenience routes for the web UI, including function listing,
spec retrieval, and the main HTML interface.

### Functions

| [`add_ui_routes`](#uf.routes.add_ui_routes)(app, function_specs, \*[, ...])   | Add UI routes to a qh app.               |
|--------------------------------------------------------------------------------------------------|------------------------------------------|
| [`create_function_handler`](#uf.routes.create_function_handler)(func, func_name)        | Create a request handler for a function. |

### uf.routes.add_ui_routes(app, function_specs, , page_title='Function Interface', custom_css=None, rjsf_theme='default')

Add UI routes to a qh app.

Adds the following routes:

- GET / : Main UI page (HTML)
- GET /api/functions : List available functions (JSON)
- GET /api/functions/{name}/spec : Get RJSF spec for function (JSON)

* **Parameters:**
  * **app** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The qh/bottle/fastapi app to add routes to
  * **function_specs** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)) – FunctionSpecStore with function specifications
  * **page_title** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Title for the web interface
  * **custom_css** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Optional custom CSS
  * **rjsf_theme** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – RJSF theme to use
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### NOTE
This function detects whether the app is using Bottle or FastAPI
and adds routes accordingly.

### uf.routes.create_function_handler(func, func_name)

Create a request handler for a function.

This wraps the function to handle HTTP requests and responses.

* **Parameters:**
  * **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – The function to wrap
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the function
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)
* **Returns:**
  A handler function compatible with web frameworks
