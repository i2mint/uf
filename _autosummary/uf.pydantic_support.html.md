# uf.pydantic_support

Pydantic integration for uf.

Provides seamless integration with Pydantic models, automatically
generating forms with validation from Pydantic models.

### Functions

| [`create_pydantic_spec`](#uf.pydantic_support.create_pydantic_spec)(func)                 | Create RJSF spec from function with Pydantic parameters.   |
|---------------------------------------------------------------------------------------------|------------------------------------------------------------|
| [`dict_to_pydantic`](#uf.pydantic_support.dict_to_pydantic)(data, model_class)        | Convert dictionary to Pydantic model instance.             |
| [`extract_field_validators`](#uf.pydantic_support.extract_field_validators)(model_class)      | Extract field validators from Pydantic model.              |
| [`extract_pydantic_params`](#uf.pydantic_support.extract_pydantic_params)(func)              | Extract Pydantic model parameters from function signature. |
| [`function_uses_pydantic`](#uf.pydantic_support.function_uses_pydantic)(func)               | Check if a function uses Pydantic models in its signature. |
| [`get_pydantic_registry`](#uf.pydantic_support.get_pydantic_registry)()                    | Get the global Pydantic registry.                          |
| [`is_pydantic_model`](#uf.pydantic_support.is_pydantic_model)(obj)                     | Check if an object is a Pydantic model.                    |
| [`pydantic_error_to_user_friendly`](#uf.pydantic_support.pydantic_error_to_user_friendly)(error)     | Convert Pydantic ValidationError to user-friendly format.  |
| [`pydantic_model_to_json_schema`](#uf.pydantic_support.pydantic_model_to_json_schema)(model_class) | Convert a Pydantic model to JSON Schema.                   |
| [`pydantic_to_dict`](#uf.pydantic_support.pydantic_to_dict)(obj)                      | Convert Pydantic model instance to dictionary.             |
| [`wrap_pydantic_function`](#uf.pydantic_support.wrap_pydantic_function)(func)               | Wrap a function that uses Pydantic models.                 |

### Classes

| [`PydanticRegistry`](#uf.pydantic_support.PydanticRegistry)()   | Registry for Pydantic models used in uf.   |
|-----------------------------------------------------------------------|--------------------------------------------|

### *class* uf.pydantic_support.PydanticRegistry

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Registry for Pydantic models used in uf.

Tracks models and provides utilities for working with them.

#### get(name)

Get a registered model by name.

* **Parameters:**
  **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model name
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
* **Returns:**
  Pydantic model class or None

#### list_models()

List all registered model names.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  List of model name strings

#### register(name, model_class)

Register a Pydantic model.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name to register under
  * **model_class** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Pydantic model class
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### uf.pydantic_support.create_pydantic_spec(func)

Create RJSF spec from function with Pydantic parameters.

* **Parameters:**
  **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function that uses Pydantic models
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]
* **Returns:**
  RJSF specification dictionary or None

### Example

```pycon
>>> from pydantic import BaseModel
>>> class UserCreate(BaseModel):
...     email: str
...     age: int
>>> def create_user(user: UserCreate):
...     pass
>>> spec = create_pydantic_spec(create_user)
```

### uf.pydantic_support.dict_to_pydantic(data, model_class)

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

### uf.pydantic_support.extract_field_validators(model_class)

Extract field validators from Pydantic model.

* **Parameters:**
  **model_class** – Pydantic model class
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary mapping field names to validator info

### uf.pydantic_support.extract_pydantic_params(func)

Extract Pydantic model parameters from function signature.

* **Parameters:**
  **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to analyze
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
* **Returns:**
  Dictionary mapping parameter names to Pydantic model classes

### uf.pydantic_support.function_uses_pydantic(func)

Check if a function uses Pydantic models in its signature.

* **Parameters:**
  **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to check
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if any parameter is a Pydantic model

### uf.pydantic_support.get_pydantic_registry()

Get the global Pydantic registry.

* **Return type:**
  [`PydanticRegistry`](#uf.pydantic_support.PydanticRegistry)
* **Returns:**
  Global PydanticRegistry instance

### uf.pydantic_support.is_pydantic_model(obj)

Check if an object is a Pydantic model.

* **Parameters:**
  **obj** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Object to check
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if object is a Pydantic BaseModel

### uf.pydantic_support.pydantic_error_to_user_friendly(error)

Convert Pydantic ValidationError to user-friendly format.

* **Parameters:**
  **error** – Pydantic ValidationError
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary with field-level errors

### uf.pydantic_support.pydantic_model_to_json_schema(model_class)

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

### uf.pydantic_support.pydantic_to_dict(obj)

Convert Pydantic model instance to dictionary.

* **Parameters:**
  **obj** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Pydantic model instance
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary representation

### uf.pydantic_support.wrap_pydantic_function(func)

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
>>> wrapped(user={'name': 'Alice'})  # Pass a dict, not a User
{'name': 'Alice'}
```
