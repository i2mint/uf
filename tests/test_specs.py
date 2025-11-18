"""Tests for uf.specs module."""

import pytest
from uf.specs import FunctionSpecStore


def sample_add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


def sample_greet(name: str, greeting: str = "Hello") -> str:
    """Greet a person."""
    return f"{greeting}, {name}!"


def test_function_spec_store_creation():
    """Test creating a FunctionSpecStore."""
    funcs = [sample_add, sample_greet]
    store = FunctionSpecStore(funcs)

    assert len(store) == 2
    assert 'sample_add' in store
    assert 'sample_greet' in store


def test_function_spec_store_getitem():
    """Test getting specs from the store."""
    funcs = [sample_add]
    store = FunctionSpecStore(funcs)

    spec = store['sample_add']

    assert 'schema' in spec
    assert 'uiSchema' in spec
    assert 'func' in spec
    assert spec['func'] == sample_add


def test_function_spec_schema_basic():
    """Test that basic schema is generated correctly."""
    funcs = [sample_add]
    store = FunctionSpecStore(funcs)

    spec = store['sample_add']
    schema = spec['schema']

    # Check schema structure
    assert schema['type'] == 'object'
    assert 'properties' in schema
    assert 'x' in schema['properties']
    assert 'y' in schema['properties']


def test_function_spec_required_params():
    """Test that required parameters are identified."""
    funcs = [sample_greet]
    store = FunctionSpecStore(funcs)

    spec = store['sample_greet']
    schema = spec['schema']

    # 'name' is required, 'greeting' has default so is optional
    assert 'name' in schema.get('required', [])
    assert 'greeting' not in schema.get('required', [])


def test_function_spec_store_iteration():
    """Test iterating over the store."""
    funcs = [sample_add, sample_greet]
    store = FunctionSpecStore(funcs)

    names = list(store)
    assert 'sample_add' in names
    assert 'sample_greet' in names


def test_function_spec_store_missing_function():
    """Test accessing a non-existent function."""
    funcs = [sample_add]
    store = FunctionSpecStore(funcs)

    with pytest.raises(KeyError):
        _ = store['nonexistent_function']


def test_function_list():
    """Test the function_list property."""
    funcs = [sample_add, sample_greet]
    store = FunctionSpecStore(funcs)

    func_list = store.function_list

    assert len(func_list) == 2
    assert any(f['name'] == 'sample_add' for f in func_list)
    assert any(f['name'] == 'sample_greet' for f in func_list)


def test_get_func():
    """Test getting the original function."""
    funcs = [sample_add]
    store = FunctionSpecStore(funcs)

    func = store.get_func('sample_add')
    assert func == sample_add
    assert func(10, 20) == 30
