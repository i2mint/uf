# uf - UI Fast

**Minimal-boilerplate web UIs for Python functions**

`uf` bridges functions → HTTP services (via [qh](https://github.com/i2mint/qh)) → Web UI forms (via [ju.rjsf](https://github.com/i2mint/ju)), following the "convention over configuration" philosophy.

## Features

- **One-line app creation**: Just pass your functions to `mk_rjsf_app()`
- **Automatic form generation**: RJSF forms created from function signatures
- **Type-aware**: Uses type hints to generate appropriate form fields
- **Zero configuration required**: Sensible defaults for everything
- **Progressive enhancement**: Customize only what you need
- **Mapping-based interfaces**: Access specs and configs as dictionaries
- **Framework agnostic**: Works with Bottle and FastAPI
- **UI decorators**: Rich metadata via `@ui_config`, `@group`, etc.
- **Function grouping**: Organize functions into categories
- **Field customization**: Configure widgets, validation, and interactions
- **Custom type support**: Register transformations for any Python type
- **Field dependencies**: Conditional display and dynamic forms
- **Testing utilities**: Built-in tools for testing your apps

## Installation

```bash
pip install uf
```

## Quick Start

```python
from uf import mk_rjsf_app

def add(x: int, y: int) -> int:
    """Add two numbers"""
    return x + y

def greet(name: str) -> str:
    """Greet a person"""
    return f"Hello, {name}!"

# Create the app
app = mk_rjsf_app([add, greet])

# Run it (for Bottle apps)
app.run(host='localhost', port=8080)
```

Then open http://localhost:8080 in your browser!

## How It Works

`uf` combines three powerful packages from the i2mint ecosystem:

1. **[qh](https://github.com/i2mint/qh)**: Converts functions → HTTP endpoints
2. **[ju.rjsf](https://github.com/i2mint/ju)**: Generates JSON Schema & RJSF specs from signatures
3. **[i2](https://github.com/i2mint/i2)**: Provides signature introspection and manipulation

The result: A complete web UI with zero boilerplate!

## Table of Contents

- [Basic Usage](#basic-usage)
- [UI Decorators](#ui-decorators)
- [Field Configuration](#field-configuration)
- [Function Grouping](#function-grouping)
- [Custom Types](#custom-types)
- [Field Dependencies](#field-dependencies)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Examples](#examples)

## Basic Usage

### Simple Example

```python
from uf import mk_rjsf_app

def multiply(x: float, y: float) -> float:
    """Multiply two numbers"""
    return x * y

app = mk_rjsf_app([multiply], page_title="Calculator")
```

### Object-Oriented Interface

For more control, use the `UfApp` class:

```python
from uf import UfApp

def fibonacci(n: int) -> list:
    """Generate Fibonacci sequence"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]

    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

# Create app
uf_app = UfApp([fibonacci])

# Call functions programmatically
result = uf_app.call('fibonacci', n=10)

# Access specs
spec = uf_app.get_spec('fibonacci')

# List available functions
functions = uf_app.list_functions()

# Run the server
uf_app.run(host='localhost', port=8080)
```

## UI Decorators

Add rich metadata to your functions using decorators:

### `@ui_config` - Complete UI Configuration

```python
from uf import ui_config, RjsfFieldConfig, get_field_config

@ui_config(
    title="User Registration",
    description="Create a new user account",
    group="Admin",
    icon="user-plus",
    order=1,
    fields={
        'email': get_field_config('email'),
        'bio': get_field_config('multiline_text'),
    }
)
def register_user(email: str, name: str, bio: str = ''):
    """Register a new user."""
    return {'email': email, 'name': name, 'bio': bio}
```

### `@group` - Simple Grouping

```python
from uf import group

@group("Admin")
def delete_user(user_id: int):
    """Delete a user from the system."""
    pass
```

### `@field_config` - Field-Level Configuration

```python
from uf import field_config, get_field_config

@field_config(
    email=get_field_config('email'),
    message=get_field_config('multiline_text'),
)
def send_message(email: str, message: str):
    """Send a message to a user."""
    pass
```

### `@hidden` - Hide from UI

```python
from uf import hidden

@hidden
def internal_function():
    """This won't appear in the UI but is accessible via API."""
    pass
```

### `@with_example` - Provide Test Data

```python
from uf import with_example

@with_example(x=10, y=20)
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y
```

### Other Decorators

```python
from uf import deprecated, requires_auth, rate_limit

@deprecated("Use new_function instead")
def old_function():
    pass

@requires_auth(roles=['admin'], permissions=['user:delete'])
def delete_user(user_id: int):
    pass

@rate_limit(calls=10, period=60)  # 10 calls per minute
def send_email(to: str, subject: str):
    pass
```

## Field Configuration

### Predefined Field Configurations

```python
from uf import get_field_config

# Available configurations:
email_config = get_field_config('email')
password_config = get_field_config('password')
url_config = get_field_config('url')
multiline_config = get_field_config('multiline_text')
long_text_config = get_field_config('long_text')
date_config = get_field_config('date')
datetime_config = get_field_config('datetime')
color_config = get_field_config('color')
file_config = get_field_config('file')
```

### Custom Field Configuration

```python
from uf import RjsfFieldConfig

custom_field = RjsfFieldConfig(
    widget='select',
    enum=['option1', 'option2', 'option3'],
    placeholder='Choose an option',
    description='Please select one option',
)

@field_config(status=custom_field)
def update_status(status: str):
    pass
```

### Field Configuration Builder

```python
from uf import RjsfConfigBuilder, RjsfFieldConfig

builder = RjsfConfigBuilder()
builder.field('name', RjsfFieldConfig(placeholder='Enter name'))
builder.field('email', RjsfFieldConfig(format='email'))
builder.order(['name', 'email', 'phone'])
builder.class_names('custom-form')

spec = builder.build(base_schema)
```

## Function Grouping

### Manual Grouping

```python
from uf import FunctionGroup, mk_grouped_app

admin_group = FunctionGroup(
    name="Admin",
    funcs=[create_user, delete_user, update_user],
    description="User administration functions",
    icon="shield",
    order=1,
)

reports_group = FunctionGroup(
    name="Reports",
    funcs=[generate_report, export_data],
    description="Reporting functions",
    icon="file-text",
    order=2,
)

app = mk_grouped_app([admin_group, reports_group])
```

### Auto-Grouping by Prefix

```python
from uf import auto_group_by_prefix

# Functions named user_create, user_delete, report_generate, etc.
# will be automatically grouped into "User", "Report", etc.
organizer = auto_group_by_prefix(
    [user_create, user_delete, report_generate],
    separator="_"
)
```

### Auto-Grouping by Module

```python
from uf import auto_group_by_module

organizer = auto_group_by_module([func1, func2, func3])
```

### Auto-Grouping by Tag

```python
from uf import auto_group_by_tag

def my_function():
    pass

my_function.__uf_group__ = "Admin"

organizer = auto_group_by_tag([my_function])
```

## Custom Types

Register custom type transformations for seamless JSON serialization:

### Using the Global Registry

```python
from uf import register_type
from pathlib import Path
from decimal import Decimal

# Register Path type
register_type(
    Path,
    to_json=str,
    from_json=Path,
    json_schema_type='string'
)

# Register Decimal type
register_type(
    Decimal,
    to_json=float,
    from_json=Decimal,
    json_schema_type='number'
)
```

### Using a Custom Registry

```python
from uf import InputTransformRegistry

registry = InputTransformRegistry()

registry.register_type(
    MyCustomType,
    to_json=lambda x: x.to_dict(),
    from_json=MyCustomType.from_dict,
    ui_widget='textarea',
    json_schema_type='object'
)

# Use with mk_rjsf_app
input_trans = registry.mk_input_trans_for_funcs([my_func])
output_trans = registry.mk_output_trans()

app = mk_rjsf_app(
    [my_func],
    input_trans=input_trans,
    output_trans=output_trans
)
```

### Pre-registered Types

The following types are automatically supported:
- `datetime.datetime`
- `datetime.date`
- `datetime.time`
- `pathlib.Path`
- `uuid.UUID`
- `decimal.Decimal`

## Field Dependencies

Create dynamic forms where fields show/hide based on other field values:

### Simple Dependency

```python
from uf import FieldDependency, DependencyAction, with_dependencies

@with_dependencies(
    FieldDependency(
        source_field='reason',
        target_field='other_reason',
        condition=lambda v: v == 'other',
        action=DependencyAction.SHOW,
    )
)
def submit_feedback(reason: str, other_reason: str = ''):
    """Submit feedback with conditional 'other' field."""
    pass
```

### Dependency Builder

```python
from uf import DependencyBuilder

builder = DependencyBuilder()
builder.when('age').greater_than(18).enable('alcohol_consent')
builder.when('country').equals('US').show('state')
builder.when('priority').in_list(['high', 'urgent']).require('manager_approval')

dependencies = builder.build()
```

### Available Actions

- `DependencyAction.SHOW` - Show the field
- `DependencyAction.HIDE` - Hide the field
- `DependencyAction.ENABLE` - Enable the field
- `DependencyAction.DISABLE` - Disable the field
- `DependencyAction.REQUIRE` - Make the field required
- `DependencyAction.OPTIONAL` - Make the field optional

## Testing

Built-in testing utilities for your uf apps:

### Test Client

```python
from uf import UfTestClient

client = UfTestClient(app)

# List functions
functions = client.list_functions()

# Get spec
spec = client.get_spec('my_function')

# Call function
result = client.call_function('my_function', {'x': 10, 'y': 20})
assert result['success']
assert result['result'] == 30
```

### Test Context Manager

```python
from uf import UfAppTester

with UfAppTester(app) as tester:
    result = tester.submit_form('add', {'x': 10, 'y': 20})
    tester.assert_success(result)
    tester.assert_result_equals(result, 30)
```

### Testing Individual Functions

```python
from uf import test_ui_function

def add(x: int, y: int) -> int:
    return x + y

# Test with expected output
test_ui_function(add, {'x': 10, 'y': 20}, expected_output=30)

# Test with expected exception
test_ui_function(
    divide,
    {'x': 10, 'y': 0},
    expected_exception=ZeroDivisionError
)
```

### Form Data Builder

```python
from uf import FormDataBuilder

form_data = (
    FormDataBuilder()
    .field('name', 'John Doe')
    .field('email', 'john@example.com')
    .fields(age=30, city='NYC')
    .build()
)
```

### Schema Assertions

```python
from uf import (
    assert_valid_rjsf_spec,
    assert_has_field,
    assert_field_type,
    assert_field_required,
)

spec = app.function_specs['my_function']

assert_valid_rjsf_spec(spec)
assert_has_field(spec, 'email')
assert_field_type(spec, 'age', 'integer')
assert_field_required(spec, 'name')
```

## Customization

### Custom CSS

```python
CUSTOM_CSS = """
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
"""

app = mk_rjsf_app(
    [func1, func2],
    page_title="My Custom App",
    custom_css=CUSTOM_CSS,
)
```

### Advanced qh Configuration

```python
from qh import AppConfig

qh_config = AppConfig(
    cors=True,
    log_requests=True,
)

app = mk_rjsf_app(
    [my_func],
    config=qh_config,
    input_trans=my_input_transformer,
    output_trans=my_output_transformer,
)
```

## Examples

See the `examples/` directory for complete working examples:

- `basic_example.py`: Simple math and text functions
- `advanced_example.py`: Customization and object-oriented interface
- `full_featured_example.py`: **Complete showcase of all features**

## API Reference

### Core Functions

#### `mk_rjsf_app(funcs, **kwargs)`

Main entry point for creating a web app from functions.

**Parameters:**
- `funcs`: Iterable of callable functions
- `config`: Optional qh.AppConfig for HTTP configuration
- `input_trans`: Optional input transformation function
- `output_trans`: Optional output transformation function
- `rjsf_config`: Optional RJSF configuration dict
- `ui_schema_factory`: Optional callable for custom UI schemas
- `page_title`: Title for the web interface (default: "Function Interface")
- `custom_css`: Optional custom CSS string
- `rjsf_theme`: RJSF theme name (default: "default")
- `add_ui`: Whether to add UI routes (default: True)
- `**qh_kwargs`: Additional arguments passed to qh.mk_app

**Returns:** Configured web application (Bottle or FastAPI)

#### `mk_grouped_app(groups, **kwargs)`

Create a uf app with grouped function navigation.

**Parameters:**
- `groups`: Iterable of FunctionGroup objects
- `**kwargs`: Same as mk_rjsf_app

**Returns:** Configured web application with grouped navigation

### Classes

#### `UfApp(funcs, **kwargs)`

Object-oriented wrapper for uf applications.

**Methods:**
- `run(host, port, **kwargs)`: Run the web server
- `call(func_name, **kwargs)`: Call a function by name
- `get_spec(func_name)`: Get RJSF spec for a function
- `list_functions()`: List all function names

#### `FunctionSpecStore(funcs, **kwargs)`

Mapping-based interface to function specifications.

#### `RjsfFieldConfig(**kwargs)`

Configuration for individual form fields.

**Attributes:**
- `widget`: Widget type
- `format`: JSON Schema format
- `enum`: List of allowed values
- `placeholder`: Placeholder text
- `description`: Help text
- And more...

#### `FunctionGroup(name, funcs, **kwargs)`

Group of functions with metadata.

**Attributes:**
- `name`: Group name
- `funcs`: List of functions
- `description`: Group description
- `icon`: Icon identifier
- `order`: Display order

#### `InputTransformRegistry()`

Registry for custom type transformations.

**Methods:**
- `register_type(py_type, **kwargs)`: Register a type
- `mk_input_trans_for_funcs(funcs)`: Create input transformation
- `mk_output_trans()`: Create output transformation

#### `FieldDependency(**kwargs)`

Define a dependency between form fields.

#### `DependencyBuilder()`

Fluent interface for building field dependencies.

### Decorators

- `@ui_config(...)`: Add complete UI configuration
- `@group(name)`: Assign to a group
- `@hidden`: Hide from UI
- `@field_config(**fields)`: Configure specific fields
- `@with_example(**kwargs)`: Attach example data
- `@deprecated(message)`: Mark as deprecated
- `@requires_auth(...)`: Mark as requiring authentication
- `@rate_limit(calls, period)`: Add rate limit metadata

### Testing Utilities

- `UfTestClient(app)`: Test client for uf apps
- `UfAppTester(app)`: Context manager for testing
- `test_ui_function(func, params, **kwargs)`: Test individual functions
- `FormDataBuilder()`: Build test form data
- `assert_valid_rjsf_spec(spec)`: Assert spec is valid
- `assert_has_field(spec, name)`: Assert field exists
- `assert_field_type(spec, name, type)`: Assert field type
- `assert_field_required(spec, name)`: Assert field is required

## Architecture

`uf` follows these design principles:

1. **Convention over Configuration**: Works out-of-the-box with sensible defaults
2. **Mapping-based Interfaces**: Access everything as dictionaries
3. **Lazy Evaluation**: Generate specs only when needed
4. **Composition over Inheritance**: Extend via decorators and transformations
5. **Progressive Enhancement**: Start simple, customize as needed

## Development Roadmap

### ✅ Milestone 1: MVP (Completed)
- [x] Core `mk_rjsf_app` function
- [x] FunctionSpecStore for spec management
- [x] HTML template generation
- [x] Essential API routes

### ✅ Milestone 2: Configuration (Completed)
- [x] RJSF customization layer
- [x] Input transformation registry
- [x] Custom field widgets

### ✅ Milestone 3: Enhancement (Completed)
- [x] Function grouping and organization
- [x] UI metadata decorators (`@ui_config`)
- [x] Auto-grouping utilities

### ✅ Milestone 4: Advanced (Completed)
- [x] Field dependencies and interactions
- [x] Testing utilities
- [x] Comprehensive examples

## Dependencies

- `qh`: HTTP service generation
- `ju`: RJSF form generation and JSON utilities
- `i2`: Signature introspection
- `dol`: Mapping interfaces
- `meshed`: Function composition utilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Related Projects

- [qh](https://github.com/i2mint/qh): HTTP services from functions
- [ju](https://github.com/i2mint/ju): JSON Schema and RJSF utilities
- [i2](https://github.com/i2mint/i2): Signature introspection
- [dol](https://github.com/i2mint/dol): Mapping interfaces
- [meshed](https://github.com/i2mint/meshed): Function composition

## Authors

Part of the [i2mint](https://github.com/i2mint) ecosystem.
