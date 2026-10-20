# uf.rjsf_config

RJSF customization layer for uf.

Provides configuration classes and builders for customizing RJSF form
generation beyond the defaults, including field widgets, UI options,
and validation rules.

### Functions

| [`apply_field_configs`](#uf.rjsf_config.apply_field_configs)(schema, ui_schema, ...)   | Apply field configurations to existing schemas.   |
|------------------------------------------------------------------------------------------------|---------------------------------------------------|
| [`get_field_config`](#uf.rjsf_config.get_field_config)(config_name)                 | Get a predefined field configuration by name.     |

### Classes

| [`ConditionalFieldConfig`](#uf.rjsf_config.ConditionalFieldConfig)(field_name, \*, condition)   | Configuration for conditional field display.            |
|------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| [`RjsfConfigBuilder`](#uf.rjsf_config.RjsfConfigBuilder)()                                 | Builder for RJSF configurations with sensible defaults. |
| [`RjsfFieldConfig`](#uf.rjsf_config.RjsfFieldConfig)([widget, ui_options, ...])          | Configuration for individual form fields.               |

### *class* uf.rjsf_config.ConditionalFieldConfig(field_name, , condition, then_schema=None, else_schema=None)

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

### *class* uf.rjsf_config.RjsfConfigBuilder

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Builder for RJSF configurations with sensible defaults.

This class helps construct RJSF specifications by providing a
fluent interface for configuring fields.

### Example

```pycon
>>> builder = RjsfConfigBuilder()
>>> _ = builder.field('email', RjsfFieldConfig(format='email'))
>>> _ = builder.field('message', RjsfFieldConfig(widget='textarea'))
>>> base_schema = {'type': 'object', 'properties': {'email': {'type': 'string'}}}
>>> spec = builder.build(base_schema)
>>> spec['schema']['properties']['email']['format']
'email'
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
  [`RjsfConfigBuilder`](#uf.rjsf_config.RjsfConfigBuilder)
* **Returns:**
  Self for method chaining

#### field(param_name, config)

Configure a specific field.

* **Parameters:**
  * **param_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the parameter/field
  * **config** ([`RjsfFieldConfig`](#uf.rjsf_config.RjsfFieldConfig)) – Configuration for the field
* **Return type:**
  [`RjsfConfigBuilder`](#uf.rjsf_config.RjsfConfigBuilder)
* **Returns:**
  Self for method chaining

#### order(field_order)

Set the order of fields in the form.

* **Parameters:**
  **field_order** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – List of field names in desired order
* **Return type:**
  [`RjsfConfigBuilder`](#uf.rjsf_config.RjsfConfigBuilder)
* **Returns:**
  Self for method chaining

### *class* uf.rjsf_config.RjsfFieldConfig(widget=None, ui_options=dict(), format=None, enum=None, description=None, placeholder=None, title=None, disabled=False, readonly=False, hidden=False, default=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Configuration for individual form fields.

This class allows fine-grained control over how form fields are
rendered in the RJSF interface.

#### widget

Widget type (e.g., ‘textarea’, ‘select’, ‘radio’, ‘date’)

* **Type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### ui_options

Additional UI options for the widget

* **Type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

#### format

JSON Schema format (e.g., ‘email’, ‘uri’, ‘date-time’)

* **Type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### enum

List of allowed values (for dropdowns)

* **Type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)]

#### description

Field description/help text

* **Type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### placeholder

Placeholder text for the input

* **Type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### title

Custom title for the field

* **Type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### disabled

Whether the field is disabled

* **Type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### readonly

Whether the field is read-only

* **Type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### hidden

Whether to hide the field

* **Type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### default

Default value for the field

* **Type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

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

### uf.rjsf_config.apply_field_configs(schema, ui_schema, field_configs)

Apply field configurations to existing schemas.

* **Parameters:**
  * **schema** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – JSON Schema to modify
  * **ui_schema** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – UI Schema to modify
  * **field_configs** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`RjsfFieldConfig`](#uf.rjsf_config.RjsfFieldConfig)]) – Mapping of field names to configurations
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]
* **Returns:**
  Tuple of (modified_schema, modified_ui_schema)

### Example

```pycon
>>> schema = {'type': 'object', 'properties': {'email': {'type': 'string'}}}
>>> configs = {'email': get_field_config('email')}
>>> schema, ui_schema = apply_field_configs(schema, {}, configs)
>>> schema['properties']['email']['format']
'email'
>>> ui_schema['email']['ui:widget']
'email'
```

### uf.rjsf_config.get_field_config(config_name)

Get a predefined field configuration by name.

* **Parameters:**
  **config_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the configuration (e.g., ‘email’, ‘multiline_text’)
* **Return type:**
  [`RjsfFieldConfig`](#uf.rjsf_config.RjsfFieldConfig)
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
