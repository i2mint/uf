# uf.templates

HTML template generation for uf web interface.

Provides functions to generate the web UI that lists functions and displays
RJSF forms for interacting with them.

### Functions

| [`generate_error_page`](#uf.templates.generate_error_page)(error_message[, status_code])   | Generate a simple error page.                     |
|------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| [`generate_index_html`](#uf.templates.generate_index_html)(function_specs, \*[, ...])      | Generate HTML page with RJSF forms for functions. |

### uf.templates.generate_error_page(error_message, status_code=500)

Generate a simple error page.

* **Parameters:**
  * **error_message** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Error message to display
  * **status_code** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – HTTP status code
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  HTML string for error page

### uf.templates.generate_index_html(function_specs, , page_title='Function Interface', custom_css=None, rjsf_theme='default', base_url='')

Generate HTML page with RJSF forms for functions.

Creates a single-page application with:

- Function list/navigation sidebar
- RJSF form for selected function
- Result display area
- Uses React and RJSF from CDN (no build step required)

* **Parameters:**
  * **function_specs** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)) – Mapping from function names to their specs
  * **page_title** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Title for the HTML page
  * **custom_css** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional custom CSS to inject
  * **rjsf_theme** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – RJSF theme to use (‘default’, ‘material-ui’, ‘semantic-ui’)
  * **base_url** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Base URL for API endpoints
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Complete HTML string for the web interface
