# uf.field_interactions

Field dependencies and interactions for uf.

Provides tools for defining relationships between form fields, such as
conditional display, dynamic validation, and field dependencies.

### Functions

| [`add_field_dependencies`](#uf.field_interactions.add_field_dependencies)(func, dependencies)   | Augment RJSF spec with field dependencies.            |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------|
| [`get_field_dependencies`](#uf.field_interactions.get_field_dependencies)(func)                 | Get field dependencies from a function.               |
| [`with_dependencies`](#uf.field_interactions.with_dependencies)(\*dependencies)            | Decorator to attach field dependencies to a function. |

### Classes

| [`DependencyAction`](#uf.field_interactions.DependencyAction)(\*values)                       | Actions that can be triggered by field dependencies.   |
|---------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| [`DependencyBuilder`](#uf.field_interactions.DependencyBuilder)()                              | Builder for creating field dependencies.               |
| [`FieldDependency`](#uf.field_interactions.FieldDependency)(source_field, target_field, ...) | Define a dependency between form fields.               |

### *class* uf.field_interactions.DependencyAction(\*values)

Bases: [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

Actions that can be triggered by field dependencies.

### *class* uf.field_interactions.DependencyBuilder

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
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`FieldDependency`](#uf.field_interactions.FieldDependency)]
* **Returns:**
  List of FieldDependency objects

#### custom(condition)

Condition: custom callable.

* **Parameters:**
  **condition** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)]) – Function that takes value and returns bool
* **Return type:**
  [`DependencyBuilder`](#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### disable(target_field)

Action: disable target field.

* **Parameters:**
  **target_field** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of field to disable
* **Return type:**
  [`DependencyBuilder`](#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### enable(target_field)

Action: enable target field.

* **Parameters:**
  **target_field** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of field to enable
* **Return type:**
  [`DependencyBuilder`](#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### equals(value)

Condition: field equals value.

* **Parameters:**
  **value** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Value to compare against
* **Return type:**
  [`DependencyBuilder`](#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### greater_than(value)

Condition: field is greater than value.

* **Parameters:**
  **value** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Value to compare against
* **Return type:**
  [`DependencyBuilder`](#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### hide(target_field)

Action: hide target field.

* **Parameters:**
  **target_field** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of field to hide
* **Return type:**
  [`DependencyBuilder`](#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### in_list(values)

Condition: field value is in list.

* **Parameters:**
  **values** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)) – List of acceptable values
* **Return type:**
  [`DependencyBuilder`](#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### is_falsy()

Condition: field is falsy.

* **Return type:**
  [`DependencyBuilder`](#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### is_truthy()

Condition: field is truthy.

* **Return type:**
  [`DependencyBuilder`](#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### less_than(value)

Condition: field is less than value.

* **Parameters:**
  **value** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Value to compare against
* **Return type:**
  [`DependencyBuilder`](#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### make_optional(target_field)

Action: make target field optional.

* **Parameters:**
  **target_field** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of field to make optional
* **Return type:**
  [`DependencyBuilder`](#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### not_equals(value)

Condition: field does not equal value.

* **Parameters:**
  **value** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Value to compare against
* **Return type:**
  [`DependencyBuilder`](#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### require(target_field)

Action: make target field required.

* **Parameters:**
  **target_field** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of field to require
* **Return type:**
  [`DependencyBuilder`](#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### show(target_field)

Action: show target field.

* **Parameters:**
  **target_field** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of field to show
* **Return type:**
  [`DependencyBuilder`](#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

#### when(field_name)

Start a dependency condition on a field.

* **Parameters:**
  **field_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the source field
* **Return type:**
  [`DependencyBuilder`](#uf.field_interactions.DependencyBuilder)
* **Returns:**
  Self for method chaining

### *class* uf.field_interactions.FieldDependency(source_field, target_field, condition, action, else_action=None)

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
  [`DependencyAction`](#uf.field_interactions.DependencyAction)
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

### uf.field_interactions.add_field_dependencies(func, dependencies)

Augment RJSF spec with field dependencies.

* **Parameters:**
  * **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to augment
  * **dependencies** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`FieldDependency`](#uf.field_interactions.FieldDependency)]) – List of field dependencies
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary with augmented schema and uiSchema

### Example

```pycon
>>> deps = [
...     FieldDependency('reason', 'other_reason',
...                    lambda v: v == 'other',
...                    DependencyAction.SHOW)
... ]
>>> spec = add_field_dependencies(my_func, deps)
```

### uf.field_interactions.get_field_dependencies(func)

Get field dependencies from a function.

* **Parameters:**
  **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to get dependencies from
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`FieldDependency`](#uf.field_interactions.FieldDependency)]
* **Returns:**
  List of FieldDependency objects

### uf.field_interactions.with_dependencies(\*dependencies)

Decorator to attach field dependencies to a function.

* **Parameters:**
  **\*dependencies** ([`FieldDependency`](#uf.field_interactions.FieldDependency)) – FieldDependency objects
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
