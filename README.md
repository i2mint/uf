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

## Usage

### Basic Example

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

### Customization

```python
from uf import mk_rjsf_app

# Custom CSS
CUSTOM_CSS = """
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
"""

app = mk_rjsf_app(
    [func1, func2, func3],
    page_title="My Custom App",
    custom_css=CUSTOM_CSS,
    rjsf_theme="default",  # or "material-ui", "semantic-ui"
)
```

### Advanced Configuration

```python
from uf import mk_rjsf_app
from qh import AppConfig

# Configure qh behavior
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

## API Reference

### `mk_rjsf_app(funcs, **kwargs)`

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

### `UfApp(funcs, **mk_rjsf_app_kwargs)`

Object-oriented wrapper for uf applications.

**Methods:**
- `run(host, port, **kwargs)`: Run the web server
- `call(func_name, **kwargs)`: Call a function by name
- `get_spec(func_name)`: Get RJSF spec for a function
- `list_functions()`: List all function names

**Attributes:**
- `app`: The underlying qh/Bottle/FastAPI app
- `function_specs`: FunctionSpecStore for metadata
- `funcs`: Dictionary of functions

### `FunctionSpecStore(funcs, **kwargs)`

Mapping-based interface to function specifications.

Provides lazy-loaded RJSF specs for functions.

## Architecture

`uf` follows these design principles:

1. **Convention over Configuration**: Works out-of-the-box with sensible defaults
2. **Mapping-based Interfaces**: Access everything as dictionaries
3. **Lazy Evaluation**: Generate specs only when needed
4. **Composition over Inheritance**: Extend via decorators and transformations
5. **Progressive Enhancement**: Start simple, customize as needed

## Roadmap

### Milestone 1: MVP ✅ (Completed)
- [x] Core `mk_rjsf_app` function
- [x] FunctionSpecStore for spec management
- [x] HTML template generation
- [x] Essential API routes

### Milestone 2: Configuration (Planned)
- [ ] RJSF customization layer
- [ ] Input transformation registry
- [ ] Custom field widgets

### Milestone 3: Enhancement (Planned)
- [ ] Function grouping and organization
- [ ] UI metadata decorators (`@ui_config`)
- [ ] Enhanced documentation generation

### Milestone 4: Advanced (Planned)
- [ ] Field dependencies and interactions
- [ ] Testing utilities
- [ ] OpenAPI integration

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
