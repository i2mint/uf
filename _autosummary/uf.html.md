# uf

uf - UI Fast: Minimal-boilerplate web UIs for Python functions.

uf bridges functions to HTTP services (via qh) to Web UI forms (via ju.rjsf),
following the “convention over configuration” philosophy.

Basic usage:

```pycon
>>> from uf import mk_rjsf_app
>>>
>>> def add(x: int, y: int) -> int:
...     '''Add two numbers'''
...     return x + y
>>>
>>> app = mk_rjsf_app([add])
>>> # app.run()  # Start the web server
```

The main entry points are:

- `mk_rjsf_app`: Create a web app from functions (functional interface)
- `UfApp`: Object-oriented wrapper with additional conveniences
- `FunctionSpecStore`: Manage function specifications (advanced usage)

Advanced features:

- `ui_config`: Decorator for UI metadata
- `RjsfFieldConfig`: Field configuration class
- `FunctionGroup`: Group functions for organization
- `InputTransformRegistry`: Custom type transformations
- Field dependencies and interactions
- Testing utilities

### Functions

| [`mk_rjsf_app`](#uf.mk_rjsf_app)(funcs, \*[, config, input_trans, ...])   | Create an RJSF-backed web app from a list of functions.    |
|-------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| [`get_field_config`](#uf.get_field_config)(config_name)                        | Get a predefined field configuration by name.              |
| [`apply_field_configs`](#uf.apply_field_configs)(schema, ui_schema, ...)          | Apply field configurations to existing schemas.            |
| [`register_type`](#uf.register_type)(\*args, \*\*kwargs)                    | Register a type in the global registry.                    |
| [`get_global_registry`](#uf.get_global_registry)()                                | Get the global transformation registry.                    |
| [`mk_grouped_app`](#uf.mk_grouped_app)(groups, \*\*mk_rjsf_app_kwargs)       | Create a uf app with grouped function navigation.          |
| [`auto_group_by_prefix`](#uf.auto_group_by_prefix)(funcs[, separator])             | Automatically group functions by name prefix.              |
| [`auto_group_by_module`](#uf.auto_group_by_module)(funcs)                          | Automatically group functions by their module.             |
| [`auto_group_by_tag`](#uf.auto_group_by_tag)(funcs[, tag_attr])                 | Automatically group functions by a tag attribute.          |
| [`ui_config`](#uf.ui_config)(\*[, title, description, group, ...])      | Decorator to add UI configuration to functions.            |
| [`group`](#uf.group)(group_name)                                    | Decorator to assign a function to a group.                 |
| [`hidden`](#uf.hidden)(func)                                         | Decorator to hide a function from the UI.                  |
| [`field_config`](#uf.field_config)(\*\*field_configs)                      | Decorator to configure specific fields of a function.      |
| [`with_example`](#uf.with_example)(\*example_args, \*\*example_kwargs)     | Decorator to attach example arguments to a function.       |
| [`deprecated`](#uf.deprecated)([message])                                | Decorator to mark a function as deprecated.                |
| [`requires_auth`](#uf.requires_auth)(\*[, roles, permissions])              | Decorator to mark a function as requiring authentication.  |
| [`rate_limit`](#uf.rate_limit)(calls, period)                            | Decorator to mark a function with rate limiting metadata.  |
| [`get_ui_config`](#uf.get_ui_config)(func)                                  | Get UI configuration from a function.                      |
| [`get_group`](#uf.get_group)(func)                                      | Get group name from a function.                            |
| [`get_field_configs`](#uf.get_field_configs)(func)                              | Get field configurations from a function.                  |
| [`is_hidden`](#uf.is_hidden)(func)                                      | Check if a function is hidden from UI.                     |
| [`get_example`](#uf.get_example)(func)                                    | Get example arguments from a function.                     |
| [`with_dependencies`](#uf.with_dependencies)(\*dependencies)                    | Decorator to attach field dependencies to a function.      |
| [`get_field_dependencies`](#uf.get_field_dependencies)(func)                         | Get field dependencies from a function.                    |
| [`test_ui_function`](#uf.test_ui_function)(func, test_inputs, \*[, ...])       | Test a function with form-like input.                      |
| [`assert_valid_rjsf_spec`](#uf.assert_valid_rjsf_spec)(spec)                         | Assert that a specification is valid RJSF format.          |
| [`assert_has_field`](#uf.assert_has_field)(spec, field_name)                   | Assert that a spec has a specific field.                   |
| [`assert_field_type`](#uf.assert_field_type)(spec, field_name, ...)             | Assert that a field has the expected type.                 |
| [`assert_field_required`](#uf.assert_field_required)(spec, field_name)              | Assert that a field is required.                           |
| [`get_global_renderer_registry`](#uf.get_global_renderer_registry)()                       | Get the global renderer registry.                          |
| [`register_renderer`](#uf.register_renderer)(renderer[, priority])              | Register a renderer in the global registry.                |
| [`render_result`](#uf.render_result)(result[, renderer_type])               | Render a result using the global registry.                 |
| [`result_renderer`](#uf.result_renderer)(renderer_type)                       | Decorator to specify result renderer for a function.       |
| [`get_result_renderer`](#uf.get_result_renderer)(func)                            | Get the result renderer type for a function.               |
| [`is_async_function`](#uf.is_async_function)(func)                              | Check if a function is async.                              |
| [`async_to_sync`](#uf.async_to_sync)(async_func)                            | Convert an async function to a synchronous wrapper.        |
| [`make_sync_compatible`](#uf.make_sync_compatible)(funcs)                          | Convert any async functions in the list to sync wrappers.  |
| [`timeout_async`](#uf.timeout_async)(seconds)                               | Decorator to add timeout to async functions.               |
| [`retry_async`](#uf.retry_async)([max_attempts, delay])                   | Decorator to add retry logic to async functions.           |
| [`is_pydantic_model`](#uf.is_pydantic_model)(obj)                               | Check if an object is a Pydantic model.                    |
| [`pydantic_model_to_json_schema`](#uf.pydantic_model_to_json_schema)(model_class)           | Convert a Pydantic model to JSON Schema.                   |
| [`function_uses_pydantic`](#uf.function_uses_pydantic)(func)                         | Check if a function uses Pydantic models in its signature. |
| [`wrap_pydantic_function`](#uf.wrap_pydantic_function)(func)                         | Wrap a function that uses Pydantic models.                 |
| [`pydantic_to_dict`](#uf.pydantic_to_dict)(obj)                                | Convert Pydantic model instance to dictionary.             |
| [`dict_to_pydantic`](#uf.dict_to_pydantic)(data, model_class)                  | Convert dictionary to Pydantic model instance.             |
| [`get_global_history_manager`](#uf.get_global_history_manager)()                         | Get the global history manager.                            |
| [`enable_history`](#uf.enable_history)(func[, max_size])                     | Decorator to enable call history for a function.           |
| [`require_auth`](#uf.require_auth)(backend[, roles, permissions])          | Decorator to require authentication for a function.        |
| [`get_global_auth_backend`](#uf.get_global_auth_backend)()                            | Get the global authentication backend.                     |
| [`cached`](#uf.cached)([ttl, backend, key_func])                     | Decorator to cache function results.                       |
| [`get_global_cache_backend`](#uf.get_global_cache_backend)()                           | Get the global cache backend.                              |
| [`background`](#uf.background)([queue_name, task_queue])                 | Decorator to run function in background.                   |
| [`get_global_task_queue`](#uf.get_global_task_queue)([name])                        | Get a global task queue by name.                           |
| [`generate_openapi_spec`](#uf.generate_openapi_spec)(funcs[, title, ...])           | Generate OpenAPI 3.0 specification.                        |
| [`add_openapi_routes`](#uf.add_openapi_routes)(app, funcs, \*\*spec_kwargs)      | Add OpenAPI routes to an app.                              |
| [`webhook`](#uf.webhook)([on, url, manager])                          | Decorator to add webhooks to a function.                   |
| [`get_global_webhook_manager`](#uf.get_global_webhook_manager)()                         | Get the global webhook manager.                            |
| [`get_theme`](#uf.get_theme)(name)                                      | Get a built-in theme by name.                              |

### Classes

| [`UfApp`](#uf.UfApp)(funcs, \*\*mk_rjsf_app_kwargs)              | Wrapper class for uf applications.                                    |
|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| [`FunctionSpecStore`](#uf.FunctionSpecStore)(funcs, \*[, rjsf_config, ...])  | A mapping from function names to their RJSF specifications.           |
| [`RjsfFieldConfig`](#uf.RjsfFieldConfig)([widget, ui_options, ...])        | Configuration for individual form fields.                             |
| [`RjsfConfigBuilder`](#uf.RjsfConfigBuilder)()                               | Builder for RJSF configurations with sensible defaults.               |
| [`ConditionalFieldConfig`](#uf.ConditionalFieldConfig)(field_name, \*, condition) | Configuration for conditional field display.                          |
| [`InputTransformRegistry`](#uf.InputTransformRegistry)()                          | Registry for custom type transformations.                             |
| [`FunctionGroup`](#uf.FunctionGroup)(name[, funcs, description, ...])    | Group of functions with metadata.                                     |
| [`FunctionOrganizer`](#uf.FunctionOrganizer)()                               | Organize functions into groups and hierarchies.                       |
| [`FieldDependency`](#uf.FieldDependency)(source_field, target_field, ...)  | Define a dependency between form fields.                              |
| [`DependencyAction`](#uf.DependencyAction)(\*values)                        | Actions that can be triggered by field dependencies.                  |
| [`DependencyBuilder`](#uf.DependencyBuilder)()                               | Builder for creating field dependencies.                              |
| [`UfTestClient`](#uf.UfTestClient)(app)                                 | Test client for uf applications.                                      |
| [`UfAppTester`](#uf.UfAppTester)(app)                                  | Context manager for testing uf apps.                                  |
| [`FormDataBuilder`](#uf.FormDataBuilder)()                                 | Builder for constructing form data for tests.                         |
| [`ResultRenderer`](#uf.ResultRenderer)()                                  | Base class for result renderers.                                      |
| [`ResultRendererRegistry`](#uf.ResultRendererRegistry)()                          | Registry for result renderers.                                        |
| [`AsyncFunctionWrapper`](#uf.AsyncFunctionWrapper)(func)                        | Wrapper for async functions that provides both sync and async access. |
| [`FunctionCall`](#uf.FunctionCall)(func_name, params[, result, ...])    | Record of a single function call.                                     |
| [`CallHistory`](#uf.CallHistory)([max_size])                           | Manage history of function calls.                                     |
| [`Preset`](#uf.Preset)(name, func_name, params[, description])    | A saved parameter preset for a function.                              |
| [`PresetManager`](#uf.PresetManager)()                                   | Manage parameter presets for functions.                               |
| [`HistoryManager`](#uf.HistoryManager)([max_history])                     | Combined manager for history and presets.                             |
| [`User`](#uf.User)(username, password_hash[, roles, ...])       | User account information.                                             |
| [`AuthBackend`](#uf.AuthBackend)()                                     | Base authentication backend.                                          |
| [`DictAuthBackend`](#uf.DictAuthBackend)()                                 | Simple in-memory dictionary-based authentication.                     |
| [`SessionManager`](#uf.SessionManager)(secret_key[, session_timeout])     | Manage user sessions.                                                 |
| [`ApiKey`](#uf.ApiKey)(key, name[, permissions, expires_at])      | API key for programmatic access.                                      |
| [`ApiKeyManager`](#uf.ApiKeyManager)([key_prefix])                       | Manage API keys for programmatic access.                              |
| [`CacheBackend`](#uf.CacheBackend)()                                    | Base class for cache backends.                                        |
| [`MemoryCache`](#uf.MemoryCache)([default_ttl, max_size])              | In-memory cache backend.                                              |
| [`DiskCache`](#uf.DiskCache)([cache_dir, default_ttl])               | Disk-based cache backend using pickle.                                |
| [`Task`](#uf.Task)(func[, args, kwargs, task_id])               | Represents a background task.                                         |
| [`TaskStatus`](#uf.TaskStatus)(\*values)                              | Status of a background task.                                          |
| [`TaskQueue`](#uf.TaskQueue)([num_workers, max_queue_size])          | FIFO queue for background tasks.                                      |
| [`OpenAPIConfig`](#uf.OpenAPIConfig)([title, version, description, ...]) | Configuration for OpenAPI generation.                                 |
| [`WebhookEvent`](#uf.WebhookEvent)(event_type, func_name, params)       | Represents a webhook event.                                           |
| [`WebhookClient`](#uf.WebhookClient)(url[, headers, timeout, ...])       | Client for sending webhooks.                                          |
| [`WebhookManager`](#uf.WebhookManager)()                                  | Manage webhooks for multiple URLs and events.                         |
| [`Theme`](#uf.Theme)(name[, colors, fonts, css])                 | UI theme configuration.                                               |
| [`ThemeConfig`](#uf.ThemeConfig)([default_theme, allow_toggle, ...])   | Configuration for theme system.                                       |

### *class* uf.ApiKey(key, name, permissions=None, expires_at=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

API key for programmatic access.

#### key

The API key string

#### name

Descriptive name

#### permissions

List of allowed permissions

#### created_at

When key was created

#### expires_at

Optional expiration

#### is_active

Whether key is active

#### has_permission(permission)

Check if key has permission.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### is_expired()

Check if key is expired.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### *class* uf.ApiKeyManager(key_prefix='sk_')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Manage API keys for programmatic access.

### Example

```pycon
>>> api_keys = ApiKeyManager()
>>> key = api_keys.create_key('mobile_app', permissions=['read'])
>>> print(f"Your API key: {key.key}")
>>> # Later, validate
>>> if api_keys.validate_key(key.key, 'read'):
...     # Allow access
```

#### create_key(name, permissions=None, expires_in_days=None)

Create a new API key.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Descriptive name for the key
  * **permissions** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – List of allowed permissions
  * **expires_in_days** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Optional expiration in days
* **Return type:**
  [`ApiKey`](uf.auth.html.md#uf.auth.ApiKey)
* **Returns:**
  Created ApiKey object

#### list_keys()

List all API keys.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`ApiKey`](uf.auth.html.md#uf.auth.ApiKey)]
* **Returns:**
  List of ApiKey objects

#### revoke_key(key)

Revoke an API key.

* **Parameters:**
  **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – API key string
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if revoked

#### validate_key(key, permission=None)

Validate an API key.

* **Parameters:**
  * **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – API key string
  * **permission** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional permission to check
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if valid

### *class* uf.AsyncFunctionWrapper(func)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Wrapper for async functions that provides both sync and async access.

This allows the same function to be called either way, depending on
the context.

### Example

```pycon
>>> async def fetch_user(user_id: int):
...     # async database query
...     return user_data
>>>
>>> wrapper = AsyncFunctionWrapper(fetch_user)
>>> # Sync call
>>> user = wrapper.call_sync(user_id=123)
>>> # Async call
>>> user = await wrapper.call_async(user_id=123)
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

### *class* uf.AuthBackend

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Base authentication backend.

Subclass this to create custom authentication backends.

#### authenticate(username, password)

Authenticate a user.

* **Parameters:**
  * **username** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Username
  * **password** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Password
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`User`](uf.auth.html.md#uf.auth.User)]
* **Returns:**
  User object if authenticated, None otherwise

#### create_user(username, password, roles=None, permissions=None, \*\*metadata)

Create a new user.

* **Parameters:**
  * **username** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Username
  * **password** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Plain text password
  * **roles** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – List of roles
  * **permissions** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – List of permissions
  * **\*\*metadata** – Additional user metadata
* **Return type:**
  [`User`](uf.auth.html.md#uf.auth.User)
* **Returns:**
  Created User object

#### delete_user(username)

Delete a user.

* **Parameters:**
  **username** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Username
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if deleted successfully

#### get_user(username)

Get a user by username.

* **Parameters:**
  **username** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Username
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`User`](uf.auth.html.md#uf.auth.User)]
* **Returns:**
  User object or None

#### update_user(username, \*\*updates)

Update user information.

* **Parameters:**
  * **username** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Username
  * **\*\*updates** – Fields to update
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if updated successfully

### *class* uf.CacheBackend

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Base class for cache backends.

#### clear()

Clear all cache entries.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### delete(key)

Delete a key from cache.

* **Parameters:**
  **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Cache key
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if deleted

#### exists(key)

Check if key exists in cache.

* **Parameters:**
  **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Cache key
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if exists and not expired

#### get(key)

Get a value from cache.

* **Parameters:**
  **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Cache key
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
* **Returns:**
  Cached value or None if not found/expired

#### set(key, value, ttl=None)

Set a value in cache.

* **Parameters:**
  * **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Cache key
  * **value** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Value to cache
  * **ttl** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Time to live in seconds
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### *class* uf.CallHistory(max_size=100)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Manage history of function calls.

### Example

```pycon
>>> history = CallHistory(max_size=100)
>>> history.record('add', {'x': 10, 'y': 20}, result=30)
>>> recent = history.get_recent('add', limit=5)
```

#### clear(func_name=None)

Clear history.

* **Parameters:**
  **func_name** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Function to clear, or None for all
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *classmethod* from_dict(data, max_size=100)

Create from dictionary.

* **Parameters:**
  * **data** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Dictionary with history data
  * **max_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum size per function
* **Return type:**
  [`CallHistory`](uf.history.html.md#uf.history.CallHistory)
* **Returns:**
  CallHistory instance

#### get_recent(func_name, limit=10)

Get recent calls for a function.

* **Parameters:**
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
  * **limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number to return
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`FunctionCall`](uf.history.html.md#uf.history.FunctionCall)]
* **Returns:**
  List of recent FunctionCall objects (newest first)

#### get_successful_calls(func_name, limit=10)

Get recent successful calls.

* **Parameters:**
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
  * **limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number to return
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`FunctionCall`](uf.history.html.md#uf.history.FunctionCall)]
* **Returns:**
  List of successful FunctionCall objects

#### record(func_name, params, result=None, success=True, error=None)

Record a function call.

* **Parameters:**
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the function
  * **params** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Parameters used
  * **result** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Result returned
  * **success** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether call succeeded
  * **error** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Error message if failed
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### to_dict()

Convert history to dictionary.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary representation

### *class* uf.ConditionalFieldConfig(field_name, , condition, then_schema=None, else_schema=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Configuration for conditional field display.

Allows fields to be shown/hidden based on the values of other fields.

### Example

```pycon
>>> # Show 'other_reason' field only when reason is 'other'
>>> config = ConditionalFieldConfig(
...     'other_reason',
...     condition={'reason': {'const': 'other'}}
... )
```

#### to_json_schema()

Convert to JSON Schema if/then/else structure.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  JSON Schema conditional structure

### *class* uf.DependencyAction(\*values)

Bases: [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

Actions that can be triggered by field dependencies.

### *class* uf.DependencyBuilder

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Builder for creating field dependencies.

Provides a fluent interface for defining dependencies between fields.

### Example

```pycon
>>> builder = DependencyBuilder()
>>> builder.when('reason').equals('other').show('other_reason')
>>> builder.when('age').greater_than(18).enable('alcohol_consent')
>>> dependencies = builder.build()
```

#### build()

Build and return the list of dependencies.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`FieldDependency`](uf.field_interactions.html.md#uf.field_interactions.FieldDependency)]
* **Returns:**
  List of FieldDependency objects

#### custom(condition)

Condition: custom callable.

* **Parameters:**
  **condition** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)]) – Function that takes value and returns bool
* **Return type:**
  [`DependencyBuilder`](uf.field_interactions.html.md#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### disable(target_field)

Action: disable target field.

* **Parameters:**
  **target_field** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of field to disable
* **Return type:**
  [`DependencyBuilder`](uf.field_interactions.html.md#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### enable(target_field)

Action: enable target field.

* **Parameters:**
  **target_field** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of field to enable
* **Return type:**
  [`DependencyBuilder`](uf.field_interactions.html.md#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### equals(value)

Condition: field equals value.

* **Parameters:**
  **value** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Value to compare against
* **Return type:**
  [`DependencyBuilder`](uf.field_interactions.html.md#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### greater_than(value)

Condition: field is greater than value.

* **Parameters:**
  **value** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Value to compare against
* **Return type:**
  [`DependencyBuilder`](uf.field_interactions.html.md#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### hide(target_field)

Action: hide target field.

* **Parameters:**
  **target_field** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of field to hide
* **Return type:**
  [`DependencyBuilder`](uf.field_interactions.html.md#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### in_list(values)

Condition: field value is in list.

* **Parameters:**
  **values** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)) – List of acceptable values
* **Return type:**
  [`DependencyBuilder`](uf.field_interactions.html.md#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### is_falsy()

Condition: field is falsy.

* **Return type:**
  [`DependencyBuilder`](uf.field_interactions.html.md#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### is_truthy()

Condition: field is truthy.

* **Return type:**
  [`DependencyBuilder`](uf.field_interactions.html.md#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### less_than(value)

Condition: field is less than value.

* **Parameters:**
  **value** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Value to compare against
* **Return type:**
  [`DependencyBuilder`](uf.field_interactions.html.md#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### make_optional(target_field)

Action: make target field optional.

* **Parameters:**
  **target_field** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of field to make optional
* **Return type:**
  [`DependencyBuilder`](uf.field_interactions.html.md#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### not_equals(value)

Condition: field does not equal value.

* **Parameters:**
  **value** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Value to compare against
* **Return type:**
  [`DependencyBuilder`](uf.field_interactions.html.md#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### require(target_field)

Action: make target field required.

* **Parameters:**
  **target_field** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of field to require
* **Return type:**
  [`DependencyBuilder`](uf.field_interactions.html.md#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### show(target_field)

Action: show target field.

* **Parameters:**
  **target_field** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of field to show
* **Return type:**
  [`DependencyBuilder`](uf.field_interactions.html.md#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### when(field_name)

Start a dependency condition on a field.

* **Parameters:**
  **field_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the source field
* **Return type:**
  [`DependencyBuilder`](uf.field_interactions.html.md#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

### *class* uf.DictAuthBackend

Bases: [`AuthBackend`](uf.auth.html.md#uf.auth.AuthBackend)

Simple in-memory dictionary-based authentication.

Suitable for development and simple applications.

### Example

```pycon
>>> backend = DictAuthBackend()
>>> backend.create_user('admin', 'secret', roles=['admin'])
>>> user = backend.authenticate('admin', 'secret')
```

#### authenticate(username, password)

Authenticate a user.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`User`](uf.auth.html.md#uf.auth.User)]

#### create_user(username, password, roles=None, permissions=None, \*\*metadata)

Create a new user.

* **Return type:**
  [`User`](uf.auth.html.md#uf.auth.User)

#### delete_user(username)

Delete a user.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### *classmethod* from_dict(users_data)

Create backend from dictionary.

* **Parameters:**
  **users_data** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Dictionary mapping usernames to user info
* **Return type:**
  [`DictAuthBackend`](uf.auth.html.md#uf.auth.DictAuthBackend)

### Example

```pycon
>>> backend = DictAuthBackend.from_dict({
...     'admin': {'password': 'secret', 'roles': ['admin']},
...     'user': {'password': 'pass', 'roles': ['user']},
... })
```

#### get_user(username)

Get a user by username.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`User`](uf.auth.html.md#uf.auth.User)]

#### update_user(username, \*\*updates)

Update user information.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### *class* uf.DiskCache(cache_dir='.uf_cache', default_ttl=3600)

Bases: [`CacheBackend`](uf.caching.html.md#uf.caching.CacheBackend)

Disk-based cache backend using pickle.

Persists cache to disk, suitable for larger datasets or persistence
across restarts.

### Example

```pycon
>>> cache = DiskCache(cache_dir='/tmp/uf_cache')
>>> cache.set('expensive_result', big_data)
```

#### clear()

Clear all cache entries.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### delete(key)

Delete a key from cache.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### get(key)

Get a value from cache.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

#### set(key, value, ttl=None)

Set a value in cache.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### *class* uf.FieldDependency(source_field, target_field, condition, action, else_action=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Define a dependency between form fields.

#### source_field

Name of the field that triggers the dependency

#### target_field

Name of the field affected by the dependency

#### condition

Function that takes source value and returns bool

#### action

Action to perform when condition is true

#### else_action

Optional action when condition is false

### Example

```pycon
>>> # Show 'other_reason' only when reason is 'other'
>>> dep = FieldDependency(
...     source_field='reason',
...     target_field='other_reason',
...     condition=lambda v: v == 'other',
...     action=DependencyAction.SHOW,
...     else_action=DependencyAction.HIDE
... )
```

#### check(value)

Check the condition and return the appropriate action.

* **Parameters:**
  **value** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Value of the source field
* **Return type:**
  [`DependencyAction`](uf.field_interactions.html.md#uf.field_interactions.DependencyAction)
* **Returns:**
  The action to perform

#### to_json_schema()

Convert to JSON Schema dependencies format.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  JSON Schema dependencies structure

#### NOTE
This uses JSON Schema’s if/then/else structure for dependencies.

#### to_ui_schema()

Convert to RJSF UI Schema dependencies format.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  UI Schema dependencies structure

### *class* uf.FormDataBuilder

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
  [`FormDataBuilder`](uf.testing.html.md#uf.testing.FormDataBuilder)
* **Returns:**
  Self for method chaining

#### fields(\*\*kwargs)

Add multiple fields at once.

* **Parameters:**
  **\*\*kwargs** – Field name-value pairs
* **Return type:**
  [`FormDataBuilder`](uf.testing.html.md#uf.testing.FormDataBuilder)
* **Returns:**
  Self for method chaining

### *class* uf.FunctionCall(func_name, params, result=None, success=True, error=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Record of a single function call.

#### func_name

Name of the function

#### params

Parameters used

#### result

Result returned (if captured)

#### timestamp

When the call was made

#### success

Whether the call succeeded

#### error

Error message if failed

#### *classmethod* from_dict(data)

Create from dictionary.

* **Parameters:**
  **data** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Dictionary with call data
* **Return type:**
  [`FunctionCall`](uf.history.html.md#uf.history.FunctionCall)
* **Returns:**
  FunctionCall instance

#### to_dict()

Convert to dictionary.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary representation

### *class* uf.FunctionGroup(name, funcs=<factory>, description='', icon=None, order=0, collapsed=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Group of functions with metadata.

#### name

Name of the group

#### funcs

Functions in this group: callables (partials and callable instances
are fine; they are named as `uf.specs.name_of_func` does), or a
`{name: callable}` mapping to name them explicitly

#### description

Description of the group

#### icon

Optional icon identifier for the group

#### order

Display order (lower numbers first)

#### collapsed

Whether the group starts collapsed in UI

### Example

```pycon
>>> math_funcs = FunctionGroup(
...     'Math',
...     [add, subtract, multiply, divide],
...     description='Mathematical operations',
...     icon='calculator'
... )
```

#### add_function(func, , name=None)

Add a function to this group.

* **Parameters:**
  * **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to add
  * **name** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Explicit name to serve it under (default: its own name).
    Giving one turns `funcs` into a `{name: callable}` mapping.
* **Return type:**
  [`FunctionGroup`](uf.organization.html.md#uf.organization.FunctionGroup)
* **Returns:**
  Self for method chaining
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If a different callable already has that name in the group

#### get_function_names()

Get names of all functions in this group.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  List of function name strings

#### *property* named_funcs *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)]*

`{name: callable}` of this group, named the way the app serves them.

```pycon
>>> import functools
>>> g = FunctionGroup('G', [len, functools.partial(int, base=2)])
>>> list(g.named_funcs)
['len', 'int']
```

### *class* uf.FunctionOrganizer

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Organize functions into groups and hierarchies.

This class provides a fluent interface for building function
organization structures that can be used to generate grouped
navigation in the UI.

### Example

```pycon
>>> organizer = FunctionOrganizer()
>>> organizer.group('Admin', [user_create, user_delete], icon='shield')
>>> organizer.group('Reports', [generate_report, export_csv], icon='file')
>>> groups = organizer.get_groups()
```

#### add_to_group(group_name, func)

Add a function to an existing group.

* **Parameters:**
  * **group_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the group
  * **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to add
* **Return type:**
  [`FunctionOrganizer`](uf.organization.html.md#uf.organization.FunctionOrganizer)
* **Returns:**
  Self for method chaining
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If group doesn’t exist

#### add_ungrouped(func)

Add a function without a group.

* **Parameters:**
  **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to add
* **Return type:**
  [`FunctionOrganizer`](uf.organization.html.md#uf.organization.FunctionOrganizer)
* **Returns:**
  Self for method chaining

#### get_all_functions()

Get all functions across all groups.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]
* **Returns:**
  List of all functions

#### get_groups()

Get all groups, sorted by order.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`FunctionGroup`](uf.organization.html.md#uf.organization.FunctionGroup)]
* **Returns:**
  List of FunctionGroup objects sorted by order

#### group(name, funcs=None, , description='', icon=None, order=0, collapsed=False)

Create and add a function group.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the group
  * **funcs** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]]) – Optional functions to add to group
  * **description** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Description of the group
  * **icon** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional icon identifier
  * **order** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Display order
  * **collapsed** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to start collapsed
* **Return type:**
  [`FunctionGroup`](uf.organization.html.md#uf.organization.FunctionGroup)
* **Returns:**
  The created FunctionGroup

### Example

```pycon
>>> organizer.group(
...     'Database',
...     [save_record, load_record],
...     description='Database operations',
...     icon='database'
... )
```

#### to_dict()

Convert organization to dictionary format.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary representation suitable for JSON serialization

### Example

```pycon
>>> org_dict = organizer.to_dict()
>>> # Can be used in templates or APIs
```

### *class* uf.FunctionSpecStore(funcs, , rjsf_config=None, ui_schema_factory=None, param_to_prop_type=None, validate=False)

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
  * **validate** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If true, every function is checked and its spec generated at
    construction, so a callable uf can’t build a form for fails there, naming
    it, rather than on the first request (default false: checked on first
    access of its spec; `mk_rjsf_app` validates).

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

### *class* uf.HistoryManager(max_history=100)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Combined manager for history and presets.

Provides a unified interface for tracking calls and managing presets.

### Example

```pycon
>>> manager = HistoryManager()
>>> manager.record_call('add', {'x': 10, 'y': 20}, result=30)
>>> manager.save_preset('quick', 'add', {'x': 10, 'y': 20})
>>> recent = manager.get_recent_calls('add')
>>> presets = manager.get_presets('add')
```

#### get_preset(preset_name, func_name)

Get a preset.

* **Parameters:**
  * **preset_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of preset
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Preset`](uf.history.html.md#uf.history.Preset)]
* **Returns:**
  Preset object or None

#### get_presets(func_name)

Get all presets for a function.

* **Parameters:**
  **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Preset`](uf.history.html.md#uf.history.Preset)]
* **Returns:**
  List of Preset objects

#### get_recent_calls(func_name, limit=10)

Get recent calls.

* **Parameters:**
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
  * **limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number to return
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`FunctionCall`](uf.history.html.md#uf.history.FunctionCall)]
* **Returns:**
  List of recent FunctionCall objects

#### *classmethod* load_from_file(filepath, max_history=100)

Load from file.

* **Parameters:**
  * **filepath** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to load from
  * **max_history** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum history size
* **Return type:**
  [`HistoryManager`](uf.history.html.md#uf.history.HistoryManager)
* **Returns:**
  HistoryManager instance

#### record_call(func_name, params, result=None, success=True, error=None)

Record a function call.

* **Parameters:**
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
  * **params** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Parameters used
  * **result** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Result returned
  * **success** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether call succeeded
  * **error** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Error message if failed
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### save_preset(preset_name, func_name, params, description='')

Save a preset.

* **Parameters:**
  * **preset_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name for preset
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
  * **params** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Parameter values
  * **description** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Optional description
* **Return type:**
  [`Preset`](uf.history.html.md#uf.history.Preset)
* **Returns:**
  Created Preset object

#### save_to_file(filepath)

Save history and presets to file.

* **Parameters:**
  **filepath** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to save to
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### *class* uf.InputTransformRegistry

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Registry for custom type transformations.

Integrates with qh’s type registry and extends it for UI needs,
allowing custom types to be properly handled in both the form
interface and the HTTP service layer.

### Example

```pycon
>>> from datetime import datetime
>>> registry = InputTransformRegistry()
>>>
>>> # Register a custom type
>>> registry.register_type(
...     datetime,
...     to_json=lambda dt: dt.isoformat(),
...     from_json=lambda s: datetime.fromisoformat(s),
...     ui_widget='datetime'
... )
```

#### from_json(value, py_type)

Transform a JSON value to Python type.

* **Parameters:**
  * **value** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – JSON value to transform
  * **py_type** ([`Type`](https://docs.python.org/3/library/typing.html#typing.Type)) – Target Python type
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
* **Returns:**
  Transformed value

#### get_all_registered_types()

Get list of all registered types.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Type`](https://docs.python.org/3/library/typing.html#typing.Type)]
* **Returns:**
  List of registered Python types

#### get_handler(py_type)

Get handler for a type.

* **Parameters:**
  **py_type** ([`Type`](https://docs.python.org/3/library/typing.html#typing.Type)) – Python type to look up
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]
* **Returns:**
  Handler dict or None if not registered

#### get_ui_config(py_type)

Get UI configuration for a type.

* **Parameters:**
  **py_type** ([`Type`](https://docs.python.org/3/library/typing.html#typing.Type)) – Python type to look up
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`RjsfFieldConfig`](uf.rjsf_config.html.md#uf.rjsf_config.RjsfFieldConfig)]
* **Returns:**
  RjsfFieldConfig or None if not registered

#### mk_input_trans_for_funcs(funcs)

Create input transformation compatible with qh.

This creates a transformation function that can be passed to
qh.mk_app as the input_trans parameter.

* **Parameters:**
  **funcs** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – List of functions to create transformation for
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)
* **Returns:**
  Transformation function for qh

### Example

```pycon
>>> from uf import mk_rjsf_app
>>> registry = InputTransformRegistry()
>>> # ... register types ...
>>> input_trans = registry.mk_input_trans_for_funcs([my_func])
>>> app = mk_rjsf_app([my_func], input_trans=input_trans)
```

#### mk_output_trans()

Create output transformation for qh.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)
* **Returns:**
  Transformation function for qh output

### Example

```pycon
>>> output_trans = registry.mk_output_trans()
>>> app = mk_rjsf_app([my_func], output_trans=output_trans)
```

#### register_type(py_type, , to_json=None, from_json=None, ui_widget=None, ui_config=None, json_schema_type=None, json_schema_format=None)

Register a type with both qh and UI configuration.

* **Parameters:**
  * **py_type** ([`Type`](https://docs.python.org/3/library/typing.html#typing.Type)) – Python type to register
  * **to_json** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Function to convert Python type to JSON-serializable
  * **from_json** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Function to convert JSON to Python type
  * **ui_widget** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – RJSF widget to use for this type
  * **ui_config** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`RjsfFieldConfig`](uf.rjsf_config.html.md#uf.rjsf_config.RjsfFieldConfig)]) – Full RjsfFieldConfig for this type
  * **json_schema_type** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – JSON Schema type (e.g., ‘string’, ‘number’)
  * **json_schema_format** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – JSON Schema format (e.g., ‘date-time’, ‘email’)
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### Example

```pycon
>>> from pathlib import Path
>>> registry.register_type(
...     Path,
...     to_json=str,
...     from_json=Path,
...     ui_widget='text',
...     json_schema_type='string'
... )
```

#### to_json(value, py_type=None)

Transform a Python value to JSON-serializable form.

* **Parameters:**
  * **value** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Value to transform
  * **py_type** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Type`](https://docs.python.org/3/library/typing.html#typing.Type)]) – Optional type hint (uses type(value) if not provided)
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
* **Returns:**
  JSON-serializable value

### *class* uf.MemoryCache(default_ttl=3600, max_size=1000)

Bases: [`CacheBackend`](uf.caching.html.md#uf.caching.CacheBackend)

In-memory cache backend.

Simple dictionary-based caching suitable for single-process applications.

### Example

```pycon
>>> cache = MemoryCache(default_ttl=3600)
>>> cache.set('key', 'value', ttl=60)
>>> value = cache.get('key')
```

#### cleanup_expired()

Remove expired entries.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)
* **Returns:**
  Number of entries removed

#### clear()

Clear all cache entries.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### delete(key)

Delete a key from cache.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### get(key)

Get a value from cache.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

#### set(key, value, ttl=None)

Set a value in cache.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### stats()

Get cache statistics.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary with cache stats

### *class* uf.OpenAPIConfig(title='API', version='1.0.0', description='', servers=None, enable_swagger=True, enable_redoc=True)

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

### *class* uf.Preset(name, func_name, params, description='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A saved parameter preset for a function.

#### name

Preset name

#### func_name

Function this preset is for

#### params

Parameter values

#### description

Optional description

#### created_at

When preset was created

#### *classmethod* from_dict(data)

Create from dictionary.

* **Parameters:**
  **data** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Dictionary with preset data
* **Return type:**
  [`Preset`](uf.history.html.md#uf.history.Preset)
* **Returns:**
  Preset instance

#### to_dict()

Convert to dictionary.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary representation

### *class* uf.PresetManager

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Manage parameter presets for functions.

### Example

```pycon
>>> presets = PresetManager()
>>> presets.save('quick_add', 'add', {'x': 10, 'y': 20}, 'Quick test')
>>> preset = presets.get('quick_add', 'add')
>>> result = my_func(**preset.params)
```

#### delete(preset_name, func_name)

Delete a preset.

* **Parameters:**
  * **preset_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of preset
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if deleted, False if not found

#### *classmethod* from_dict(data)

Create from dictionary.

* **Parameters:**
  **data** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Dictionary with preset data
* **Return type:**
  [`PresetManager`](uf.history.html.md#uf.history.PresetManager)
* **Returns:**
  PresetManager instance

#### get(preset_name, func_name)

Get a preset.

* **Parameters:**
  * **preset_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of preset
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Preset`](uf.history.html.md#uf.history.Preset)]
* **Returns:**
  Preset object or None

#### list_presets(func_name)

List all presets for a function.

* **Parameters:**
  **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Preset`](uf.history.html.md#uf.history.Preset)]
* **Returns:**
  List of Preset objects

#### save(preset_name, func_name, params, description='')

Save a parameter preset.

* **Parameters:**
  * **preset_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name for this preset
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function this preset is for
  * **params** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Parameter values
  * **description** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Optional description
* **Return type:**
  [`Preset`](uf.history.html.md#uf.history.Preset)
* **Returns:**
  Created Preset object

#### to_dict()

Convert to dictionary.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary representation

### *class* uf.ResultRenderer

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

### *class* uf.ResultRendererRegistry

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Registry for result renderers.

Manages a collection of renderers and selects the appropriate one
for each result type.

#### register(renderer, priority=0)

Register a renderer.

* **Parameters:**
  * **renderer** ([`ResultRenderer`](uf.renderers.html.md#uf.renderers.ResultRenderer)) – ResultRenderer instance
  * **priority** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Higher priority renderers are tried first
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### register_for_type(result_type, renderer)

Register a renderer for a specific type.

* **Parameters:**
  * **result_type** ([`Type`](https://docs.python.org/3/library/typing.html#typing.Type)) – Python type to match
  * **renderer** ([`ResultRenderer`](uf.renderers.html.md#uf.renderers.ResultRenderer)) – ResultRenderer instance
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

### *class* uf.RjsfConfigBuilder

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Builder for RJSF configurations with sensible defaults.

This class helps construct RJSF specifications by providing a
fluent interface for configuring fields.

### Example

```pycon
>>> builder = RjsfConfigBuilder()
>>> builder.field('email', RjsfFieldConfig(format='email'))
>>> builder.field('message', RjsfFieldConfig(widget='textarea'))
>>> spec = builder.build(base_schema)
```

#### build(base_schema, base_ui_schema=None)

Build the final RJSF specification.

* **Parameters:**
  * **base_schema** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Base JSON Schema to augment
  * **base_ui_schema** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Optional base UI Schema to augment
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary with ‘schema’ and ‘uiSchema’ keys

#### class_names(class_names)

Set CSS class names for the form.

* **Parameters:**
  **class_names** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Space-separated CSS class names
* **Return type:**
  [`RjsfConfigBuilder`](uf.rjsf_config.html.md#uf.rjsf_config.RjsfConfigBuilder)
* **Returns:**
  Self for method chaining

#### field(param_name, config)

Configure a specific field.

* **Parameters:**
  * **param_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the parameter/field
  * **config** ([`RjsfFieldConfig`](uf.rjsf_config.html.md#uf.rjsf_config.RjsfFieldConfig)) – Configuration for the field
* **Return type:**
  [`RjsfConfigBuilder`](uf.rjsf_config.html.md#uf.rjsf_config.RjsfConfigBuilder)
* **Returns:**
  Self for method chaining

#### order(field_order)

Set the order of fields in the form.

* **Parameters:**
  **field_order** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – List of field names in desired order
* **Return type:**
  [`RjsfConfigBuilder`](uf.rjsf_config.html.md#uf.rjsf_config.RjsfConfigBuilder)
* **Returns:**
  Self for method chaining

### *class* uf.RjsfFieldConfig(widget=None, ui_options=<factory>, format=None, enum=None, description=None, placeholder=None, title=None, disabled=False, readonly=False, hidden=False, default=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Configuration for individual form fields.

This class allows fine-grained control over how form fields are
rendered in the RJSF interface.

#### widget

Widget type (e.g., ‘textarea’, ‘select’, ‘radio’, ‘date’)

#### ui_options

Additional UI options for the widget

#### format

JSON Schema format (e.g., ‘email’, ‘uri’, ‘date-time’)

#### enum

List of allowed values (for dropdowns)

#### description

Field description/help text

#### placeholder

Placeholder text for the input

#### title

Custom title for the field

#### disabled

Whether the field is disabled

#### readonly

Whether the field is read-only

#### hidden

Whether to hide the field

#### default

Default value for the field

### Example

```pycon
>>> email_config = RjsfFieldConfig(
...     widget='email',
...     format='email',
...     placeholder='user@example.com'
... )
```

#### to_json_schema_patch()

Convert to JSON Schema properties.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary of JSON Schema properties to merge into schema

#### to_ui_schema_patch()

Convert to RJSF UI Schema properties.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary of UI Schema properties for this field

### *class* uf.SessionManager(secret_key, session_timeout=3600)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Manage user sessions.

### Example

```pycon
>>> sessions = SessionManager(secret_key='my-secret')
>>> session_id = sessions.create_session('admin')
>>> user = sessions.get_session(session_id)
```

#### cleanup_expired()

Remove expired sessions.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)
* **Returns:**
  Number of sessions removed

#### create_session(username, data=None)

Create a new session.

* **Parameters:**
  * **username** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Username for session
  * **data** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Optional session data
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Session ID

#### delete_session(session_id)

Delete a session.

* **Parameters:**
  **session_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Session ID
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if deleted

#### get_session(session_id)

Get session data.

* **Parameters:**
  **session_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Session ID
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]
* **Returns:**
  Session data or None if expired/invalid

### *class* uf.Task(func, args=(), kwargs=None, task_id=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Represents a background task.

#### task_id

Unique task identifier

#### func_name

Name of the function

#### args

Positional arguments

#### kwargs

Keyword arguments

#### status

Current task status

#### result

Task result (if completed)

#### error

Error message (if failed)

#### created_at

When task was created

#### started_at

When task started running

#### completed_at

When task completed

#### progress

Progress percentage (0-100)

#### execute()

Execute the task.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
* **Returns:**
  Task result
* **Raises:**
  [**Exception**](https://docs.python.org/3/builtins/exceptions.html#Exception) – If task execution fails

#### to_dict()

Convert task to dictionary.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary representation

### *class* uf.TaskQueue(num_workers=1, max_queue_size=100)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

FIFO queue for background tasks.

### Example

```pycon
>>> task_queue = TaskQueue(num_workers=2)
>>> task_queue.start()
>>> task_id = task_queue.submit(expensive_function, x=10, y=20)
>>> status = task_queue.get_status(task_id)
>>> result = task_queue.get_result(task_id)
```

#### cancel_task(task_id)

Cancel a task.

* **Parameters:**
  **task_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Task ID
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if cancelled

#### NOTE
Can only cancel pending tasks

#### get_result(task_id, wait=False, timeout=None)

Get task result.

* **Parameters:**
  * **task_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Task ID
  * **wait** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to wait for completion
  * **timeout** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]) – Optional timeout in seconds
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
* **Returns:**
  Task result
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If task not found
  * [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) – If task failed
  * [**TimeoutError**](https://docs.python.org/3/builtins/exceptions.html#TimeoutError) – If wait times out

#### get_status(task_id)

Get task status.

* **Parameters:**
  **task_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Task ID
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`TaskStatus`](#uf.TaskStatus)]
* **Returns:**
  TaskStatus or None if not found

#### get_task(task_id)

Get task object.

* **Parameters:**
  **task_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Task ID
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Task`](#uf.Task)]
* **Returns:**
  Task object or None

#### queue_size()

Get current queue size.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)
* **Returns:**
  Number of pending tasks

#### start()

Start worker threads.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### stats()

Get queue statistics.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary with statistics

#### stop(wait=True)

Stop worker threads.

* **Parameters:**
  **wait** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to wait for threads to finish
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### submit(func, \*args, task_id=None, \*\*kwargs)

Submit a task for execution.

* **Parameters:**
  * **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to execute
  * **\*args** – Positional arguments
  * **task_id** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional task ID
  * **\*\*kwargs** – Keyword arguments
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Task ID
* **Raises:**
  [**Full**](https://docs.python.org/3/library/queue.html#queue.Full) – If queue is full

### *class* uf.TaskStatus(\*values)

Bases: [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

Status of a background task.

### *class* uf.Theme(name, colors=<factory>, fonts=<factory>, css='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

UI theme configuration.

#### name

Theme name

#### colors

Color palette

#### fonts

Font configuration

#### css

Additional custom CSS

#### to_css()

Convert theme to CSS.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  CSS string

### *class* uf.ThemeConfig(default_theme='light', allow_toggle=True, available_themes=None, custom_theme=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Configuration for theme system.

### Example

```pycon
>>> config = ThemeConfig(
...     default_theme='dark',
...     allow_toggle=True,
...     available_themes=['light', 'dark', 'ocean']
... )
```

#### get_css()

Get CSS for all available themes.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Combined CSS string

#### get_js()

Get JavaScript for theme functionality.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  JavaScript string

#### get_toggle_button_html()

Get HTML for theme toggle button.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  HTML string or empty if toggle not allowed

### *class* uf.UfApp(funcs, \*\*mk_rjsf_app_kwargs)

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

### *class* uf.UfAppTester(app)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Context manager for testing uf apps.

Provides a testing context with common utilities and assertions.

### Example

```pycon
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

### *class* uf.UfTestClient(app)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Test client for uf applications.

Provides a convenient interface for testing uf apps without
running a web server.

### Example

```pycon
>>> from uf import mk_rjsf_app
>>> app = mk_rjsf_app([my_func])
>>> client = UfTestClient(app)
>>> response = client.call_function('my_func', {'x': 10, 'y': 20})
>>> assert response['success']
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

### *class* uf.User(username, password_hash, roles=<factory>, permissions=<factory>, metadata=<factory>, created_at=<factory>, is_active=True)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

User account information.

#### username

Unique username

#### password_hash

Hashed password

#### roles

List of role names

#### permissions

List of permission strings

#### metadata

Additional user metadata

#### created_at

When user was created

#### is_active

Whether account is active

#### has_all_roles(roles)

Check if user has all of the given roles.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### has_any_role(roles)

Check if user has any of the given roles.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### has_permission(permission)

Check if user has a permission.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### has_role(role)

Check if user has a role.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### *class* uf.WebhookClient(url, headers=None, timeout=10.0, retry_count=3)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Client for sending webhooks.

### Example

```pycon
>>> client = WebhookClient('https://example.com/webhook')
>>> client.send(WebhookEvent('success', 'my_func', {}, result=42))
```

#### send(event, async_send=True)

Send a webhook event.

* **Parameters:**
  * **event** ([`WebhookEvent`](uf.webhooks.html.md#uf.webhooks.WebhookEvent)) – WebhookEvent to send
  * **async_send** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to send asynchronously
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if sent successfully (for sync sends)

### *class* uf.WebhookEvent(event_type, func_name, params, result=None, error=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Represents a webhook event.

#### event_type

Type of event (‘success’, ‘failure’, ‘start’)

#### func_name

Name of the function

#### params

Function parameters

#### result

Function result (if success)

#### error

Error message (if failure)

#### timestamp

When event occurred

#### to_dict()

Convert to dictionary for JSON serialization.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary representation

### *class* uf.WebhookManager

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Manage webhooks for multiple URLs and events.

### Example

```pycon
>>> manager = WebhookManager()
>>> manager.add_webhook('https://example.com/hook1', events=['success'])
>>> manager.add_webhook('https://example.com/hook2', events=['failure'])
>>> manager.trigger('success', 'my_func', {}, result=42)
```

#### add_webhook(url, events=None, headers=None, condition=None)

Add a webhook.

* **Parameters:**
  * **url** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Webhook URL
  * **events** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – List of event types to trigger on (None = all)
  * **headers** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Optional HTTP headers
  * **condition** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Optional callable to filter events
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### list_webhooks()

List all registered webhooks.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]
* **Returns:**
  List of webhook configurations

#### remove_webhook(url)

Remove a webhook by URL.

* **Parameters:**
  **url** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Webhook URL
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if removed

#### trigger(event_type, func_name, params, result=None, error=None)

Trigger webhooks for an event.

* **Parameters:**
  * **event_type** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Event type
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
  * **params** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Function parameters
  * **result** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Function result
  * **error** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Error message
* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)
* **Returns:**
  Number of webhooks triggered

### uf.add_openapi_routes(app, funcs, \*\*spec_kwargs)

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

### uf.apply_field_configs(schema, ui_schema, field_configs)

Apply field configurations to existing schemas.

* **Parameters:**
  * **schema** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – JSON Schema to modify
  * **ui_schema** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – UI Schema to modify
  * **field_configs** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`RjsfFieldConfig`](uf.rjsf_config.html.md#uf.rjsf_config.RjsfFieldConfig)]) – Mapping of field names to configurations
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]
* **Returns:**
  Tuple of (modified_schema, modified_ui_schema)

### Example

```pycon
>>> configs = {
...     'email': get_field_config('email'),
...     'bio': get_field_config('multiline_text'),
... }
>>> schema, ui_schema = apply_field_configs(schema, ui_schema, configs)
```

### uf.assert_field_required(spec, field_name)

Assert that a field is required.

* **Parameters:**
  * **spec** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – The specification
  * **field_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the field
* **Raises:**
  [**AssertionError**](https://docs.python.org/3/builtins/exceptions.html#AssertionError) – If field is not required

### uf.assert_field_type(spec, field_name, expected_type)

Assert that a field has the expected type.

* **Parameters:**
  * **spec** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – The specification
  * **field_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the field
  * **expected_type** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Expected JSON Schema type
* **Raises:**
  [**AssertionError**](https://docs.python.org/3/builtins/exceptions.html#AssertionError) – If type doesn’t match

### uf.assert_has_field(spec, field_name)

Assert that a spec has a specific field.

* **Parameters:**
  * **spec** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – The specification
  * **field_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the field to check
* **Raises:**
  [**AssertionError**](https://docs.python.org/3/builtins/exceptions.html#AssertionError) – If field not found

### uf.assert_valid_rjsf_spec(spec)

Assert that a specification is valid RJSF format.

* **Parameters:**
  **spec** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – The specification to validate
* **Raises:**
  [**AssertionError**](https://docs.python.org/3/builtins/exceptions.html#AssertionError) – If spec is invalid

### uf.async_to_sync(async_func)

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
...     # async operations
...     return data
>>> sync_fetch = async_to_sync(fetch_data)
>>> result = sync_fetch('https://example.com')
```

### uf.auto_group_by_module(funcs)

Automatically group functions by their module.

* **Parameters:**
  **funcs** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Functions to organize
* **Return type:**
  [`FunctionOrganizer`](uf.organization.html.md#uf.organization.FunctionOrganizer)
* **Returns:**
  FunctionOrganizer with module-based groups

### Example

```pycon
>>> from myapp import user_ops, report_ops
>>> funcs = [user_ops.create, user_ops.delete, report_ops.generate]
>>> organizer = auto_group_by_module(funcs)
```

### uf.auto_group_by_prefix(funcs, separator='_')

Automatically group functions by name prefix.

Groups functions based on the part of their name before the separator.
For example, with separator=”_”:

- user_create, user_delete → “user” group
- report_generate, report_export → “report” group

* **Parameters:**
  * **funcs** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Functions to organize
  * **separator** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Separator character (default: “_”)
* **Return type:**
  [`FunctionOrganizer`](uf.organization.html.md#uf.organization.FunctionOrganizer)
* **Returns:**
  FunctionOrganizer with auto-generated groups

### Example

```pycon
>>> funcs = [user_create, user_delete, report_generate, admin_reset]
>>> organizer = auto_group_by_prefix(funcs)
>>> # Creates groups: 'user', 'report', 'admin'
```

### uf.auto_group_by_tag(funcs, tag_attr='_\_uf_group_\_')

Automatically group functions by a tag attribute.

Functions can be tagged with a group name using an attribute.

* **Parameters:**
  * **funcs** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Functions to organize
  * **tag_attr** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the attribute to use for grouping
* **Return type:**
  [`FunctionOrganizer`](uf.organization.html.md#uf.organization.FunctionOrganizer)
* **Returns:**
  FunctionOrganizer with tag-based groups

### Example

```pycon
>>> def create_user(name: str):
...     pass
>>> create_user.__uf_group__ = 'Admin'
>>>
>>> organizer = auto_group_by_tag([create_user, other_func])
```

### uf.background(queue_name='default', task_queue=None)

Decorator to run function in background.

* **Parameters:**
  * **queue_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the queue to use
  * **task_queue** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`TaskQueue`](#uf.TaskQueue)]) – Optional TaskQueue instance
* **Returns:**
  Decorator function

### Example

```pycon
>>> @background()
... def send_email(to: str, subject: str):
...     # Long-running email sending
...     pass
>>>
>>> task_id = send_email('user@example.com', 'Hello')
>>> # Returns immediately with task_id
```

### uf.cached(ttl=3600, backend=None, key_func=None)

Decorator to cache function results.

* **Parameters:**
  * **ttl** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Time to live in seconds
  * **backend** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`CacheBackend`](uf.caching.html.md#uf.caching.CacheBackend)]) – Cache backend (uses global MemoryCache if None)
  * **key_func** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Optional function to generate cache key
* **Returns:**
  Decorator function

### Example

```pycon
>>> @cached(ttl=3600)
... def expensive_calculation(x: int, y: int) -> int:
...     # Only runs once per unique (x, y) combination
...     return heavy_computation(x, y)
>>>
>>> result = expensive_calculation(10, 20)  # Computes
>>> result2 = expensive_calculation(10, 20)  # From cache
```

### uf.deprecated(message=None)

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

### uf.dict_to_pydantic(data, model_class)

Convert dictionary to Pydantic model instance.

* **Parameters:**
  * **data** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Dictionary of data
  * **model_class** – Pydantic model class
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
* **Returns:**
  Pydantic model instance
* **Raises:**
  **ValidationError** – If data doesn’t match model schema

### uf.enable_history(func, max_size=100)

Decorator to enable call history for a function.

* **Parameters:**
  **max_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum history size
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)
* **Returns:**
  Decorator function

### Example

```pycon
>>> @enable_history
... def my_function(x: int):
...     return x * 2
```

### uf.field_config(\*\*field_configs)

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

### uf.function_uses_pydantic(func)

Check if a function uses Pydantic models in its signature.

* **Parameters:**
  **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to check
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if any parameter is a Pydantic model

### uf.generate_openapi_spec(funcs, title='API', version='1.0.0', description='', servers=None)

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

### uf.get_example(func)

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

### uf.get_field_config(config_name)

Get a predefined field configuration by name.

* **Parameters:**
  **config_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the configuration (e.g., ‘email’, ‘multiline_text’)
* **Return type:**
  [`RjsfFieldConfig`](uf.rjsf_config.html.md#uf.rjsf_config.RjsfFieldConfig)
* **Returns:**
  RjsfFieldConfig instance
* **Raises:**
  [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – If config_name not found

### Example

```pycon
>>> email_config = get_field_config('email')
>>> email_config.format
'email'
```

### uf.get_field_configs(func)

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

### uf.get_field_dependencies(func)

Get field dependencies from a function.

* **Parameters:**
  **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to get dependencies from
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`FieldDependency`](uf.field_interactions.html.md#uf.field_interactions.FieldDependency)]
* **Returns:**
  List of FieldDependency objects

### uf.get_global_auth_backend()

Get the global authentication backend.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`AuthBackend`](uf.auth.html.md#uf.auth.AuthBackend)]

### uf.get_global_cache_backend()

Get the global cache backend.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`CacheBackend`](uf.caching.html.md#uf.caching.CacheBackend)]

### uf.get_global_history_manager()

Get the global history manager.

* **Return type:**
  [`HistoryManager`](uf.history.html.md#uf.history.HistoryManager)
* **Returns:**
  Global HistoryManager instance

### uf.get_global_registry()

Get the global transformation registry.

* **Return type:**
  [`InputTransformRegistry`](uf.trans.html.md#uf.trans.InputTransformRegistry)
* **Returns:**
  The global InputTransformRegistry instance

### uf.get_global_renderer_registry()

Get the global renderer registry.

* **Return type:**
  [`ResultRendererRegistry`](uf.renderers.html.md#uf.renderers.ResultRendererRegistry)
* **Returns:**
  The global ResultRendererRegistry instance

### uf.get_global_task_queue(name='default')

Get a global task queue by name.

* **Parameters:**
  **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Queue name
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`TaskQueue`](#uf.TaskQueue)]
* **Returns:**
  TaskQueue or None

### uf.get_global_webhook_manager()

Get the global webhook manager.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`WebhookManager`](uf.webhooks.html.md#uf.webhooks.WebhookManager)]

### uf.get_group(func)

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

### uf.get_result_renderer(func)

Get the result renderer type for a function.

* **Parameters:**
  **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to check
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  Renderer type string or None

### uf.get_theme(name)

Get a built-in theme by name.

* **Parameters:**
  **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Theme name
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Theme`](uf.themes.html.md#uf.themes.Theme)]
* **Returns:**
  Theme object or None

### Example

```pycon
>>> theme = get_theme('dark')
>>> css = theme.to_css()
```

### uf.get_ui_config(func)

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

### uf.group(group_name)

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

### uf.hidden(func)

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

### uf.is_async_function(func)

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

### uf.is_hidden(func)

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

### uf.is_pydantic_model(obj)

Check if an object is a Pydantic model.

* **Parameters:**
  **obj** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Object to check
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if object is a Pydantic BaseModel

### uf.make_sync_compatible(funcs)

Convert any async functions in the list to sync wrappers.

* **Parameters:**
  **funcs** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – List of functions (may include async)
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]
* **Returns:**
  List of functions (all synchronous)

### Example

```pycon
>>> funcs = [sync_func, async_func, another_sync]
>>> compatible = make_sync_compatible(funcs)
>>> # All functions in compatible are now callable synchronously
```

### uf.mk_grouped_app(groups, \*\*mk_rjsf_app_kwargs)

Create a uf app with grouped function navigation.

* **Parameters:**
  * **groups** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`FunctionGroup`](uf.organization.html.md#uf.organization.FunctionGroup)]) – Iterable of FunctionGroup objects
  * **\*\*mk_rjsf_app_kwargs** – Arguments passed to mk_rjsf_app
* **Returns:**
  Configured web application with grouped navigation

### Example

```pycon
>>> admin_group = FunctionGroup('Admin', [user_create, user_delete])
>>> reports_group = FunctionGroup('Reports', [generate_report])
>>> app = mk_grouped_app([admin_group, reports_group])
```

### uf.mk_rjsf_app(funcs, , config=None, input_trans=None, output_trans=None, rjsf_config=None, ui_schema_factory=None, param_to_prop_type=None, page_title='Function Interface', function_display_names=None, custom_css=None, rjsf_theme='default', add_ui=True, \*\*qh_kwargs)

Create an RJSF-backed web app from a list of functions.

This is the main entry point for uf. It combines qh’s HTTP service
generation with ju’s RJSF form generation to create a complete
web interface for your functions.

* **Parameters:**
  * **funcs** (`Union`[[`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)], [`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]]) – Iterable of callables to expose via web UI – functions, partials,
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

### uf.pydantic_model_to_json_schema(model_class)

Convert a Pydantic model to JSON Schema.

* **Parameters:**
  **model_class** – Pydantic model class
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  JSON Schema dictionary

### Example

```pycon
>>> from pydantic import BaseModel
>>> class User(BaseModel):
...     name: str
...     age: int
>>> schema = pydantic_model_to_json_schema(User)
```

### uf.pydantic_to_dict(obj)

Convert Pydantic model instance to dictionary.

* **Parameters:**
  **obj** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Pydantic model instance
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary representation

### uf.rate_limit(calls, period)

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

### uf.register_renderer(renderer, priority=0)

Register a renderer in the global registry.

* **Parameters:**
  * **renderer** ([`ResultRenderer`](uf.renderers.html.md#uf.renderers.ResultRenderer)) – ResultRenderer instance
  * **priority** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Higher priority renderers are tried first
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### uf.register_type(\*args, \*\*kwargs)

Register a type in the global registry.

This is a convenience function that uses the global registry.
See InputTransformRegistry.register_type for full documentation.

### uf.render_result(result, renderer_type=None)

Render a result using the global registry.

* **Parameters:**
  * **result** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The function result to render
  * **renderer_type** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional specific renderer type to use
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Rendered result dictionary

### uf.require_auth(backend, roles=None, permissions=None)

Decorator to require authentication for a function.

* **Parameters:**
  * **backend** ([`AuthBackend`](uf.auth.html.md#uf.auth.AuthBackend)) – Authentication backend
  * **roles** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Required roles (any)
  * **permissions** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Required permissions (all)
* **Returns:**
  Decorator function

### Example

```pycon
>>> backend = DictAuthBackend.from_dict({
...     'admin': {'password': 'secret', 'roles': ['admin']}
... })
>>> @require_auth(backend, roles=['admin'])
... def delete_all():
...     pass
```

### uf.requires_auth(, roles=None, permissions=None)

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

### uf.result_renderer(renderer_type)

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

### uf.retry_async(max_attempts=3, delay=1.0)

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

### uf.test_ui_function(func, test_inputs, , expected_output=None, expected_exception=None)

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

### uf.timeout_async(seconds)

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

### uf.ui_config(, title=None, description=None, group=None, fields=None, hidden=False, icon=None, order=None)

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

### uf.webhook(on=None, url=None, manager=None)

Decorator to add webhooks to a function.

* **Parameters:**
  * **on** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – List of events to trigger on (‘success’, ‘failure’, ‘start’)
  * **url** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional webhook URL
  * **manager** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`WebhookManager`](uf.webhooks.html.md#uf.webhooks.WebhookManager)]) – Optional WebhookManager instance
* **Returns:**
  Decorator function

### Example

```pycon
>>> @webhook(on=['success', 'failure'])
... def process_order(order_id: int):
...     # Process order
...     return {'status': 'processed'}
```

### uf.with_dependencies(\*dependencies)

Decorator to attach field dependencies to a function.

* **Parameters:**
  **\*dependencies** ([`FieldDependency`](uf.field_interactions.html.md#uf.field_interactions.FieldDependency)) – FieldDependency objects
* **Returns:**
  Decorator function

### Example

```pycon
>>> @with_dependencies(
...     FieldDependency('reason', 'other_reason',
...                    lambda v: v == 'other',
...                    DependencyAction.SHOW)
... )
... def submit_feedback(reason: str, other_reason: str = ''):
...     pass
```

### uf.with_example(\*example_args, \*\*example_kwargs)

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

### uf.wrap_pydantic_function(func)

Wrap a function that uses Pydantic models.

The wrapper converts dict inputs to Pydantic models before calling
the function, and converts Pydantic outputs back to dicts.

* **Parameters:**
  **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to wrap
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)
* **Returns:**
  Wrapped function

### Example

```pycon
>>> from pydantic import BaseModel
>>> class User(BaseModel):
...     name: str
>>> def create_user(user: User) -> User:
...     return user
>>> wrapped = wrap_pydantic_function(create_user)
>>> result = wrapped({'name': 'Alice'})  # Pass dict, not User
```

### Modules

| [`async_support`](uf.async_support.html.md#module-uf.async_support)           | Async function support for uf.                                |
|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| [`auth`](uf.auth.html.md#module-uf.auth)                             | Authentication and authorization for uf.                      |
| [`base`](uf.base.html.md#module-uf.base)                             | Core functionality for uf - UI Fast.                          |
| [`caching`](uf.caching.html.md#module-uf.caching)                       | Result caching for uf.                                        |
| [`decorators`](uf.decorators.html.md#module-uf.decorators)                 | UI metadata decorators for uf.                                |
| [`field_interactions`](uf.field_interactions.html.md#module-uf.field_interactions) | Field dependencies and interactions for uf.                   |
| [`history`](uf.history.html.md#module-uf.history)                       | Call history and presets for uf.                              |
| [`openapi`](uf.openapi.html.md#module-uf.openapi)                       | OpenAPI/Swagger integration for uf.                           |
| [`optional`](uf.optional.html.md#module-uf.optional)                     | Import optional dependencies without masking broken installs. |
| [`organization`](uf.organization.html.md#module-uf.organization)             | Function grouping and organization for uf.                    |
| [`pydantic_support`](uf.pydantic_support.html.md#module-uf.pydantic_support)     | Pydantic integration for uf.                                  |
| [`renderers`](uf.renderers.html.md#module-uf.renderers)                   | Result rendering system for uf.                               |
| [`rjsf_config`](uf.rjsf_config.html.md#module-uf.rjsf_config)               | RJSF customization layer for uf.                              |
| [`routes`](uf.routes.html.md#module-uf.routes)                         | API routes for uf web interface.                              |
| [`specs`](uf.specs.html.md#module-uf.specs)                           | Function specification management for uf.                     |
| [`templates`](uf.templates.html.md#module-uf.templates)                   | HTML template generation for uf web interface.                |
| [`testing`](uf.testing.html.md#module-uf.testing)                       | Testing utilities for uf applications.                        |
| [`themes`](uf.themes.html.md#module-uf.themes)                         | Theme and styling support for uf.                             |
| [`trans`](uf.trans.html.md#module-uf.trans)                           | Input transformation integration for uf.                      |
| [`webhooks`](uf.webhooks.html.md#module-uf.webhooks)                     | Webhook support for uf.                                       |
