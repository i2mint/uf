# uf.renderers

Result rendering system for uf.

Provides smart rendering of function results based on their type,
including tables, charts, DataFrames, JSON, and custom renderers.

### Functions

| [`get_global_renderer_registry`](#uf.renderers.get_global_renderer_registry)()                    | Get the global renderer registry.                    |
|----------------------------------------------------------------------------------------------------|------------------------------------------------------|
| [`get_result_renderer`](#uf.renderers.get_result_renderer)(func)                         | Get the result renderer type for a function.         |
| [`register_renderer`](#uf.renderers.register_renderer)(renderer[, priority])           | Register a renderer in the global registry.          |
| [`register_renderer_for_type`](#uf.renderers.register_renderer_for_type)(result_type, renderer) | Register a renderer for a specific type.             |
| [`render_result`](#uf.renderers.render_result)(result[, renderer_type])            | Render a result using the global registry.           |
| [`result_renderer`](#uf.renderers.result_renderer)(renderer_type)                    | Decorator to specify result renderer for a function. |

### Classes

| [`ChartRenderer`](#uf.renderers.ChartRenderer)()          | Render data suitable for charts.   |
|---------------------------------------------------------------------------|------------------------------------|
| [`DataFrameRenderer`](#uf.renderers.DataFrameRenderer)()      | Render pandas DataFrame.           |
| [`HtmlRenderer`](#uf.renderers.HtmlRenderer)()           | Render HTML strings.               |
| [`ImageRenderer`](#uf.renderers.ImageRenderer)()          | Render image data.                 |
| [`JsonRenderer`](#uf.renderers.JsonRenderer)()           | Render results as formatted JSON.  |
| [`MarkdownRenderer`](#uf.renderers.MarkdownRenderer)()       | Render markdown strings.           |
| [`ResultRenderer`](#uf.renderers.ResultRenderer)()         | Base class for result renderers.   |
| [`ResultRendererRegistry`](#uf.renderers.ResultRendererRegistry)() | Registry for result renderers.     |
| [`TableRenderer`](#uf.renderers.TableRenderer)()          | Render list of dicts as a table.   |

### *class* uf.renderers.ChartRenderer

Bases: [`ResultRenderer`](#uf.renderers.ResultRenderer)

Render data suitable for charts.

#### can_render(result)

Can render list of dicts with numeric values.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### render(result)

Render as chart data.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* uf.renderers.DataFrameRenderer

Bases: [`ResultRenderer`](#uf.renderers.ResultRenderer)

Render pandas DataFrame.

#### can_render(result)

Can render pandas DataFrame.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### render(result)

Render DataFrame as table.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* uf.renderers.HtmlRenderer

Bases: [`ResultRenderer`](#uf.renderers.ResultRenderer)

Render HTML strings.

#### can_render(result)

Can render strings that look like HTML.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### render(result)

Render as HTML.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* uf.renderers.ImageRenderer

Bases: [`ResultRenderer`](#uf.renderers.ResultRenderer)

Render image data.

#### can_render(result)

Can render bytes that look like images.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### render(result)

Render image as base64.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* uf.renderers.JsonRenderer

Bases: [`ResultRenderer`](#uf.renderers.ResultRenderer)

Render results as formatted JSON.

#### can_render(result)

Can render any JSON-serializable result.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### render(result)

Render as formatted JSON.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* uf.renderers.MarkdownRenderer

Bases: [`ResultRenderer`](#uf.renderers.ResultRenderer)

Render markdown strings.

#### can_render(result)

Can render strings that look like markdown.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### render(result)

Render as markdown.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* uf.renderers.ResultRenderer

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Base class for result renderers.

Subclass this to create custom renderers for specific result types.

#### can_render(result)

Check if this renderer can handle the given result.

* **Parameters:**
  **result** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The function result to render
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if this renderer can handle the result

#### render(result)

Render the result to a displayable format.

* **Parameters:**
  **result** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The function result to render
* **Returns:**
  - ‘type’: Renderer type (e.g., ‘table’, ‘chart’, ‘json’)
  - ’data’: Rendered data
  - ’options’: Optional rendering options
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* uf.renderers.ResultRendererRegistry

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Registry for result renderers.

Manages a collection of renderers and selects the appropriate one
for each result type.

#### register(renderer, priority=0)

Register a renderer.

* **Parameters:**
  * **renderer** ([`ResultRenderer`](#uf.renderers.ResultRenderer)) – ResultRenderer instance
  * **priority** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Higher priority renderers are tried first
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### register_for_type(result_type, renderer)

Register a renderer for a specific type.

* **Parameters:**
  * **result_type** ([`Type`](https://docs.python.org/3/library/typing.html#typing.Type)) – Python type to match
  * **renderer** ([`ResultRenderer`](#uf.renderers.ResultRenderer)) – ResultRenderer instance
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### render(result, renderer_type=None)

Render a result using the appropriate renderer.

* **Parameters:**
  * **result** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The function result to render
  * **renderer_type** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional specific renderer type to use
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Rendered result dictionary

### *class* uf.renderers.TableRenderer

Bases: [`ResultRenderer`](#uf.renderers.ResultRenderer)

Render list of dicts as a table.

#### can_render(result)

Can render list of dictionaries.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### render(result)

Render as table.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### uf.renderers.get_global_renderer_registry()

Get the global renderer registry.

* **Return type:**
  [`ResultRendererRegistry`](#uf.renderers.ResultRendererRegistry)
* **Returns:**
  The global ResultRendererRegistry instance

### uf.renderers.get_result_renderer(func)

Get the result renderer type for a function.

* **Parameters:**
  **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to check
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  Renderer type string or None

### uf.renderers.register_renderer(renderer, priority=0)

Register a renderer in the global registry.

* **Parameters:**
  * **renderer** ([`ResultRenderer`](#uf.renderers.ResultRenderer)) – ResultRenderer instance
  * **priority** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Higher priority renderers are tried first
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### uf.renderers.register_renderer_for_type(result_type, renderer)

Register a renderer for a specific type.

* **Parameters:**
  * **result_type** ([`Type`](https://docs.python.org/3/library/typing.html#typing.Type)) – Python type to match
  * **renderer** ([`ResultRenderer`](#uf.renderers.ResultRenderer)) – ResultRenderer instance
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### uf.renderers.render_result(result, renderer_type=None)

Render a result using the global registry.

* **Parameters:**
  * **result** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The function result to render
  * **renderer_type** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional specific renderer type to use
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Rendered result dictionary

### uf.renderers.result_renderer(renderer_type)

Decorator to specify result renderer for a function.

* **Parameters:**
  **renderer_type** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Type of renderer to use
* **Returns:**
  Decorator function

### Example

```pycon
>>> @result_renderer('table')
... def get_users() -> list[dict]:
...     return [{'name': 'Alice', 'age': 30}]
```
