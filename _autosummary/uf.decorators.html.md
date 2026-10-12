# uf.decorators

UI metadata decorators for uf.

Provides decorators for annotating functions with UI configuration,
grouping, and field specifications.

### Functions

| [`deprecated`](#uf.decorators.deprecated)([message])                            | Decorator to mark a function as deprecated.               |
|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| [`field_config`](#uf.decorators.field_config)(\*\*field_configs)                  | Decorator to configure specific fields of a function.     |
| [`get_example`](#uf.decorators.get_example)(func)                                | Get example arguments from a function.                    |
| [`get_field_configs`](#uf.decorators.get_field_configs)(func)                          | Get field configurations from a function.                 |
| [`get_group`](#uf.decorators.get_group)(func)                                  | Get group name from a function.                           |
| [`get_ui_config`](#uf.decorators.get_ui_config)(func)                              | Get UI configuration from a function.                     |
| [`group`](#uf.decorators.group)(group_name)                                | Decorator to assign a function to a group.                |
| [`hidden`](#uf.decorators.hidden)(func)                                     | Decorator to hide a function from the UI.                 |
| [`is_hidden`](#uf.decorators.is_hidden)(func)                                  | Check if a function is hidden from UI.                    |
| [`rate_limit`](#uf.decorators.rate_limit)(calls, period)                        | Decorator to mark a function with rate limiting metadata. |
| [`requires_auth`](#uf.decorators.requires_auth)(\*[, roles, permissions])          | Decorator to mark a function as requiring authentication. |
| [`ui_config`](#uf.decorators.ui_config)(\*[, title, description, group, ...])  | Decorator to add UI configuration to functions.           |
| [`with_example`](#uf.decorators.with_example)(\*example_args, \*\*example_kwargs) | Decorator to attach example arguments to a function.      |

### uf.decorators.deprecated(message=None)

Decorator to mark a function as deprecated.

* **Parameters:**
  **message** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional deprecation message
* **Returns:**
  Decorator function

### Example

```pycon
>>> @deprecated("Use new_function instead")
... def old_function():
...     pass
```

### uf.decorators.field_config(\*\*field_configs)

Decorator to configure specific fields of a function.

* **Parameters:**
  **\*\*field_configs** ([`RjsfFieldConfig`](uf.rjsf_config.html.md#uf.rjsf_config.RjsfFieldConfig)) – Keyword arguments mapping parameter names to configs
* **Returns:**
  Decorator function

### Example

```pycon
>>> from uf.rjsf_config import get_field_config
>>>
>>> @field_config(
...     email=get_field_config('email'),
...     bio=get_field_config('multiline_text')
... )
... def create_profile(email: str, bio: str):
...     pass
```

### uf.decorators.get_example(func)

Get example arguments from a function.

* **Parameters:**
  **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to get example from
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]
* **Returns:**
  Tuple of (args, kwargs) or None if no example

### Example

```pycon
>>> example = get_example(my_function)
>>> if example:
...     args, kwargs = example
...     result = my_function(*args, **kwargs)
```

### uf.decorators.get_field_configs(func)

Get field configurations from a function.

* **Parameters:**
  **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to get field configs from
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`RjsfFieldConfig`](uf.rjsf_config.html.md#uf.rjsf_config.RjsfFieldConfig)]
* **Returns:**
  Dictionary mapping parameter names to RjsfFieldConfig

### Example

```pycon
>>> field_configs = get_field_configs(my_function)
>>> if 'email' in field_configs:
...     print(field_configs['email'].format)
```

### uf.decorators.get_group(func)

Get group name from a function.

* **Parameters:**
  **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to get group from
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  Group name or None

### Example

```pycon
>>> group = get_group(my_function)
>>> if group:
...     print(f"Function is in group: {group}")
```

### uf.decorators.get_ui_config(func)

Get UI configuration from a function.

* **Parameters:**
  **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to get config from
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]
* **Returns:**
  Configuration dictionary or None if not configured

### Example

```pycon
>>> config = get_ui_config(my_function)
>>> if config:
...     print(config['title'])
```

### uf.decorators.group(group_name)

Decorator to assign a function to a group.

Simpler alternative to ui_config when you only need to set the group.

* **Parameters:**
  **group_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the group
* **Returns:**
  Decorator function

### Example

```pycon
>>> @group("Admin")
... def delete_user(user_id: int):
...     pass
```

### uf.decorators.hidden(func)

Decorator to hide a function from the UI.

The function will still be callable via the API but won’t appear
in the UI navigation.

* **Parameters:**
  **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to hide
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)
* **Returns:**
  Decorated function

### Example

```pycon
>>> @hidden
... def internal_function():
...     pass
```

### uf.decorators.is_hidden(func)

Check if a function is hidden from UI.

* **Parameters:**
  **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to check
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if function is hidden, False otherwise

### Example

```pycon
>>> if not is_hidden(my_function):
...     # Show in UI
...     pass
```

### uf.decorators.rate_limit(calls, period)

Decorator to mark a function with rate limiting metadata.

This is metadata-only; actual rate limiting must be implemented
separately.

* **Parameters:**
  * **calls** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of calls allowed
  * **period** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Time period in seconds
* **Returns:**
  Decorator function

### Example

```pycon
>>> @rate_limit(calls=10, period=60)  # 10 calls per minute
... def send_email(to: str, subject: str):
...     pass
```

### uf.decorators.requires_auth(, roles=None, permissions=None)

Decorator to mark a function as requiring authentication.

This is metadata-only; actual authentication must be implemented
separately in the application layer.

* **Parameters:**
  * **roles** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – List of required roles
  * **permissions** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – List of required permissions
* **Returns:**
  Decorator function

### Example

```pycon
>>> @requires_auth(roles=['admin'], permissions=['user:delete'])
... def delete_user(user_id: int):
...     pass
```

### uf.decorators.ui_config(, title=None, description=None, group=None, fields=None, hidden=False, icon=None, order=None)

Decorator to add UI configuration to functions.

This decorator attaches metadata to functions that uf can use to
customize the generated UI.

* **Parameters:**
  * **title** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Custom title for the function in UI
  * **description** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Custom description (overrides docstring)
  * **group** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Group name for organization
  * **fields** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`RjsfFieldConfig`](uf.rjsf_config.html.md#uf.rjsf_config.RjsfFieldConfig)]]) – Dictionary mapping parameter names to RjsfFieldConfig
  * **hidden** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to hide this function from UI
  * **icon** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Icon identifier for the function
  * **order** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Display order within group
* **Returns:**
  Decorator function

### Example

```pycon
>>> @ui_config(
...     title="User Registration",
...     group="Admin",
...     fields={'email': RjsfFieldConfig(format='email')}
... )
... def register_user(email: str, name: str):
...     pass
```

### uf.decorators.with_example(\*example_args, \*\*example_kwargs)

Decorator to attach example arguments to a function.

This can be used to provide example/test data that appears in the UI.

* **Parameters:**
  * **\*example_args** – Example positional arguments
  * **\*\*example_kwargs** – Example keyword arguments
* **Returns:**
  Decorator function

### Example

```pycon
>>> @with_example(x=10, y=20)
... def add(x: int, y: int) -> int:
...     return x + y
```
