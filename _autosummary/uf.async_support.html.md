# uf.async_support

Async function support for uf.

Provides utilities for detecting and handling async functions,
allowing seamless integration of async def functions in uf apps.

### Functions

| [`async_to_sync`](#uf.async_support.async_to_sync)(async_func)            | Convert an async function to a synchronous wrapper.       |
|---------------------------------------------------------------------------------------|-----------------------------------------------------------|
| [`batch_async_calls`](#uf.async_support.batch_async_calls)(async_func, calls) | Execute multiple async function calls concurrently.       |
| [`create_async_handler`](#uf.async_support.create_async_handler)(async_func)     | Create a handler for async functions in web frameworks.   |
| [`detect_async_framework`](#uf.async_support.detect_async_framework)(app)          | Detect if the web framework supports async natively.      |
| [`is_async_function`](#uf.async_support.is_async_function)(func)              | Check if a function is async.                             |
| [`is_framework_async_capable`](#uf.async_support.is_framework_async_capable)(app)      | Check if the framework supports async handlers natively.  |
| [`make_sync_compatible`](#uf.async_support.make_sync_compatible)(funcs)          | Convert any async functions in the list to sync wrappers. |
| [`retry_async`](#uf.async_support.retry_async)([max_attempts, delay])   | Decorator to add retry logic to async functions.          |
| [`timeout_async`](#uf.async_support.timeout_async)(seconds)               | Decorator to add timeout to async functions.              |

### Classes

| [`AsyncContext`](#uf.async_support.AsyncContext)()             | Context manager for handling async operations in uf.                  |
|-----------------------------------------------------------------------------|-----------------------------------------------------------------------|
| [`AsyncFunctionWrapper`](#uf.async_support.AsyncFunctionWrapper)(func) | Wrapper for async functions that provides both sync and async access. |

### *class* uf.async_support.AsyncContext

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Context manager for handling async operations in uf.

This helps manage event loops and async resources properly.

### Example

```pycon
>>> async def my_async_function(param):
...     return param * 2
>>> with AsyncContext() as ctx:
...     ctx.run(my_async_function(21))
42
```

#### run(coro)

Run a coroutine in this context.

* **Parameters:**
  **coro** – Coroutine to run
* **Returns:**
  Result of the coroutine

### *class* uf.async_support.AsyncFunctionWrapper(func)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Wrapper for async functions that provides both sync and async access.

This allows the same function to be called either way, depending on
the context.

### Example

```pycon
>>> async def fetch_user(user_id: int):
...     return {'id': user_id}
>>>
>>> wrapper = AsyncFunctionWrapper(fetch_user)
>>> wrapper.call_sync(user_id=123)  # sync call
{'id': 123}
>>> import asyncio
>>> asyncio.run(wrapper.call_async(user_id=123))  # async call (use await in async code)
{'id': 123}
```

#### *async* call_async(\*args, \*\*kwargs)

Call the function asynchronously.

* **Parameters:**
  * **\*args** – Positional arguments
  * **\*\*kwargs** – Keyword arguments
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
* **Returns:**
  Function result

#### call_sync(\*args, \*\*kwargs)

Call the function synchronously.

* **Parameters:**
  * **\*args** – Positional arguments
  * **\*\*kwargs** – Keyword arguments
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
* **Returns:**
  Function result

### uf.async_support.async_to_sync(async_func)

Convert an async function to a synchronous wrapper.

* **Parameters:**
  **async_func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Async function to wrap
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)
* **Returns:**
  Synchronous wrapper function

### Example

```pycon
>>> async def fetch_data(url: str):
...     return {'url': url}
>>> sync_fetch = async_to_sync(fetch_data)
>>> sync_fetch('https://example.com')
{'url': 'https://example.com'}
```

### uf.async_support.batch_async_calls(async_func, calls)

Execute multiple async function calls concurrently.

* **Parameters:**
  * **async_func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Async function to call
  * **calls** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – List of dicts with ‘args’ and ‘kwargs’ for each call
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
* **Returns:**
  List of results in same order as calls

### Example

```pycon
>>> async def fetch_user(user_id: int):
...     return {'id': user_id}
>>>
>>> calls = [
...     {'args': (), 'kwargs': {'user_id': 1}},
...     {'args': (), 'kwargs': {'user_id': 2}},
...     {'args': (), 'kwargs': {'user_id': 3}},
... ]
>>> batch_async_calls(fetch_user, calls)
[{'id': 1}, {'id': 2}, {'id': 3}]
```

### uf.async_support.create_async_handler(async_func)

Create a handler for async functions in web frameworks.

This is useful for integrating with frameworks that may or may not
support async natively (like Bottle vs FastAPI).

* **Parameters:**
  **async_func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Async function to create handler for
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)
* **Returns:**
  Appropriate handler (async or sync wrapped)

### Example

```pycon
>>> async def my_endpoint(param: str):
...     result = await some_async_operation(param)
...     return result
>>> handler = create_async_handler(my_endpoint)
```

### uf.async_support.detect_async_framework(app)

Detect if the web framework supports async natively.

* **Parameters:**
  **app** – The web application
* **Returns:**
  ‘fastapi’, ‘bottle’, ‘unknown’
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### uf.async_support.is_async_function(func)

Check if a function is async.

* **Parameters:**
  **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to check
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if function is async

### Example

```pycon
>>> async def my_async_func():
...     pass
>>> is_async_function(my_async_func)
True
```

### uf.async_support.is_framework_async_capable(app)

Check if the framework supports async handlers natively.

* **Parameters:**
  **app** – The web application
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if framework supports async

### uf.async_support.make_sync_compatible(funcs)

Convert any async functions in the list to sync wrappers.

* **Parameters:**
  **funcs** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – List of functions (may include async)
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]
* **Returns:**
  List of functions (all synchronous)

### Example

```pycon
>>> def sync_func(): return 1
>>> async def async_func(): return 2
>>> compatible = make_sync_compatible([sync_func, async_func])
>>> [f() for f in compatible]  # all callable synchronously
[1, 2]
```

### uf.async_support.retry_async(max_attempts=3, delay=1.0)

Decorator to add retry logic to async functions.

* **Parameters:**
  * **max_attempts** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number of attempts
  * **delay** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Delay between attempts in seconds
* **Returns:**
  Decorator function

### Example

```pycon
>>> @retry_async(max_attempts=3, delay=2.0)
... async def unreliable_api_call():
...     # May fail, will retry
...     return await external_api.fetch()
```

### uf.async_support.timeout_async(seconds)

Decorator to add timeout to async functions.

* **Parameters:**
  **seconds** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Timeout in seconds
* **Returns:**
  Decorator function

### Example

```pycon
>>> @timeout_async(5.0)
... async def slow_operation():
...     await asyncio.sleep(10)  # Will timeout after 5s
```
