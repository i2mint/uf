# uf.testing

Testing utilities for uf applications.

Provides tools for testing uf apps, including test clients, fixtures,
and assertion helpers.

### Functions

| [`assert_field_required`](#uf.testing.assert_field_required)(spec, field_name)        | Assert that a field is required.                  |
|-------------------------------------------------------------------------------------------------|---------------------------------------------------|
| [`assert_field_type`](#uf.testing.assert_field_type)(spec, field_name, ...)       | Assert that a field has the expected type.        |
| [`assert_has_field`](#uf.testing.assert_has_field)(spec, field_name)             | Assert that a spec has a specific field.          |
| [`assert_valid_rjsf_spec`](#uf.testing.assert_valid_rjsf_spec)(spec)                   | Assert that a specification is valid RJSF format. |
| [`mock_function_response`](#uf.testing.mock_function_response)(app, func_name, ...)    | Context manager to mock a function's response.    |
| [`test_ui_function`](#uf.testing.test_ui_function)(func, test_inputs, \*[, ...]) | Test a function with form-like input.             |

### Classes

| [`FormDataBuilder`](#uf.testing.FormDataBuilder)()   | Builder for constructing form data for tests.   |
|----------------------------------------------------------------------|-------------------------------------------------|
| [`UfAppTester`](#uf.testing.UfAppTester)(app)    | Context manager for testing uf apps.            |
| [`UfTestClient`](#uf.testing.UfTestClient)(app)   | Test client for uf applications.                |

### *class* uf.testing.FormDataBuilder

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Builder for constructing form data for tests.

Provides a fluent interface for building test form data.

### Example

```pycon
>>> form_data = (
...     FormDataBuilder()
...     .field('name', 'John Doe')
...     .field('email', 'john@example.com')
...     .field('age', 30)
...     .build()
... )
```

#### build()

Build and return the form data.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary of form data

#### field(name, value)

Add a field to the form data.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Field name
  * **value** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Field value
* **Return type:**
  [`FormDataBuilder`](#uf.testing.FormDataBuilder)
* **Returns:**
  Self for method chaining

#### fields(\*\*kwargs)

Add multiple fields at once.

* **Parameters:**
  **\*\*kwargs** – Field name-value pairs
* **Return type:**
  [`FormDataBuilder`](#uf.testing.FormDataBuilder)
* **Returns:**
  Self for method chaining

### *class* uf.testing.UfAppTester(app)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Context manager for testing uf apps.

Provides a testing context with common utilities and assertions.

### Example

```pycon
>>> from uf import mk_rjsf_app
>>> def add(x: int, y: int) -> int:
...     return x + y
>>> app = mk_rjsf_app([add])
>>> with UfAppTester(app) as tester:
...     result = tester.submit_form('add', {'x': 10, 'y': 20})
...     tester.assert_success(result)
...     tester.assert_result_equals(result, 30)
```

#### assert_error_type(result, error_type)

Assert that error type matches.

* **Parameters:**
  * **result** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Result dictionary
  * **error_type** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Expected error type name
* **Raises:**
  [**AssertionError**](https://docs.python.org/3/builtins/exceptions.html#AssertionError) – If error types don’t match

#### assert_failure(result, message='')

Assert that a result indicates failure.

* **Parameters:**
  * **result** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Result dictionary
  * **message** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Optional custom error message
* **Raises:**
  [**AssertionError**](https://docs.python.org/3/builtins/exceptions.html#AssertionError) – If result is successful

#### assert_result_equals(result, expected)

Assert that result value equals expected.

* **Parameters:**
  * **result** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Result dictionary
  * **expected** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Expected value
* **Raises:**
  [**AssertionError**](https://docs.python.org/3/builtins/exceptions.html#AssertionError) – If values don’t match

#### assert_success(result, message='')

Assert that a result indicates success.

* **Parameters:**
  * **result** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Result dictionary
  * **message** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Optional custom error message
* **Raises:**
  [**AssertionError**](https://docs.python.org/3/builtins/exceptions.html#AssertionError) – If result is not successful

#### get_history()

Get history of all results.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]
* **Returns:**
  List of result dictionaries

#### submit_form(func_name, form_data)

Simulate form submission.

* **Parameters:**
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the function
  * **form_data** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Form data as dictionary
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Result dictionary

### *class* uf.testing.UfTestClient(app)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Test client for uf applications.

Provides a convenient interface for testing uf apps without
running a web server.

### Example

```pycon
>>> from uf import mk_rjsf_app
>>> def my_func(x: int, y: int) -> int:
...     return x + y
>>> app = mk_rjsf_app([my_func])
>>> client = UfTestClient(app)
>>> response = client.call_function('my_func', {'x': 10, 'y': 20})
>>> assert response['success']
>>> response['result']
30
```

#### call_function(func_name, params, , expect_success=True)

Call a function with the given parameters.

* **Parameters:**
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the function to call
  * **params** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Dictionary of parameters
  * **expect_success** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to expect success (raises on failure)
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Result dictionary with ‘success’ and ‘result’ or ‘error’
* **Raises:**
  [**AssertionError**](https://docs.python.org/3/builtins/exceptions.html#AssertionError) – If expect_success=True and call fails

#### get_spec(func_name)

Get RJSF specification for a function.

* **Parameters:**
  **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the function
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Function specification dict
* **Raises:**
  [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – If function not found

#### list_functions()

Get list of available function names.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  List of function name strings

#### validate_params(func_name, params)

Validate parameters against function schema.

* **Parameters:**
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the function
  * **params** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Parameters to validate
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]
* **Returns:**
  Tuple of (is_valid, error_message)

### uf.testing.assert_field_required(spec, field_name)

Assert that a field is required.

* **Parameters:**
  * **spec** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – The specification
  * **field_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the field
* **Raises:**
  [**AssertionError**](https://docs.python.org/3/builtins/exceptions.html#AssertionError) – If field is not required

### uf.testing.assert_field_type(spec, field_name, expected_type)

Assert that a field has the expected type.

* **Parameters:**
  * **spec** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – The specification
  * **field_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the field
  * **expected_type** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Expected JSON Schema type
* **Raises:**
  [**AssertionError**](https://docs.python.org/3/builtins/exceptions.html#AssertionError) – If type doesn’t match

### uf.testing.assert_has_field(spec, field_name)

Assert that a spec has a specific field.

* **Parameters:**
  * **spec** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – The specification
  * **field_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the field to check
* **Raises:**
  [**AssertionError**](https://docs.python.org/3/builtins/exceptions.html#AssertionError) – If field not found

### uf.testing.assert_valid_rjsf_spec(spec)

Assert that a specification is valid RJSF format.

* **Parameters:**
  **spec** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – The specification to validate
* **Raises:**
  [**AssertionError**](https://docs.python.org/3/builtins/exceptions.html#AssertionError) – If spec is invalid

### uf.testing.mock_function_response(app, func_name, mock_result)

Context manager to mock a function’s response.

* **Parameters:**
  * **app** – The uf application
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of function to mock
  * **mock_result** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Value to return
* **Yields:**
  None

### Example

```pycon
>>> from uf import mk_rjsf_app
>>> def get_user(id: int):
...     return {'name': 'Real'}
>>> app = mk_rjsf_app([get_user])
>>> client = UfTestClient(app)
>>> with mock_function_response(app, 'get_user', {'name': 'Test'}):
...     client.call_function('get_user', {'id': 1})['result']
{'name': 'Test'}
```

### uf.testing.test_ui_function(func, test_inputs, , expected_output=None, expected_exception=None)

Test a function with form-like input.

* **Parameters:**
  * **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to test
  * **test_inputs** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Dictionary of test parameters
  * **expected_output** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Expected return value (if any)
  * **expected_exception** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`type`](https://docs.python.org/3/builtins/functions.html#type)]) – Expected exception type (if any)
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if test passes
* **Raises:**
  [**AssertionError**](https://docs.python.org/3/builtins/exceptions.html#AssertionError) – If test fails

### Example

```pycon
>>> def add(x: int, y: int) -> int:
...     return x + y
>>> test_ui_function(add, {'x': 10, 'y': 20}, expected_output=30)
True
```
