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


# --- ju integration (i2mint/uf#4) -------------------------------------------
# These tests take the ju path with no fallback guard. If ju is not installed
# they skip with a visible reason; if ju is installed but fails to import (e.g.
# an undeclared dependency of ju), they fail with the real traceback.

import importlib
import importlib.abc
import sys
import warnings

from uf.specs import JuNotInstalledWarning, _import_func_to_form_spec


def _require_ju():
    """Import ``ju.rjsf``; skip only if ``ju`` itself is absent."""
    try:
        return importlib.import_module('ju.rjsf')
    except ModuleNotFoundError as e:
        if e.name == 'ju':
            pytest.skip(f'ju not installed ({e.name!r} missing)')
        raise


def test_ju_path_is_taken_without_fallback():
    """The spec must come from ju, not from the basic fallback."""
    _require_ju()
    with warnings.catch_warnings():
        warnings.simplefilter('error', JuNotInstalledWarning)
        spec = FunctionSpecStore([sample_add])['sample_add']

    schema = spec['schema']
    assert schema['properties'] == {
        'x': {'type': 'integer'},
        'y': {'type': 'integer'},
    }
    # The basic fallback never emits a uiSchema; ju always does.
    assert spec['uiSchema'], 'empty uiSchema: the basic fallback ran, not ju'


def test_ju_required_fields():
    """Params without defaults are required; defaulted ones are not (ju>=0.1.33)."""
    _require_ju()
    schema = FunctionSpecStore([sample_greet])['sample_greet']['schema']
    assert schema['required'] == ['name']
    assert schema['properties']['greeting'].get('default') == 'Hello'


class _JuAbsentFinder(importlib.abc.MetaPathFinder):
    """Make ``ju`` look uninstalled, exactly as a real absence does."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'ju':
            raise ModuleNotFoundError("No module named 'ju'", name='ju')
        return None


def _block_ju(monkeypatch):
    for name in [m for m in sys.modules if m == 'ju' or m.startswith('ju.')]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, 'meta_path', [_JuAbsentFinder(), *sys.meta_path])


def test_missing_ju_falls_back_with_visible_warning(monkeypatch):
    _block_ju(monkeypatch)
    with pytest.warns(JuNotInstalledWarning, match='not installed'):
        spec = FunctionSpecStore([sample_greet])['sample_greet']
    assert spec['schema']['required'] == ['name']
    assert spec['uiSchema'] == {}


class _BrokenJuRjsfFinder(importlib.abc.MetaPathFinder):
    """Simulate ju being installed but one of its own dependencies missing."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'ju.rjsf':
            raise ModuleNotFoundError("No module named 'dill'", name='dill')
        return None


def test_broken_ju_dependency_raises_instead_of_falling_back(monkeypatch):
    _require_ju()
    monkeypatch.delitem(sys.modules, 'ju.rjsf', raising=False)
    monkeypatch.setattr(sys, 'meta_path', [_BrokenJuRjsfFinder(), *sys.meta_path])
    monkeypatch.delattr(sys.modules['ju'], 'rjsf', raising=False)
    with pytest.raises(ModuleNotFoundError) as excinfo:
        _import_func_to_form_spec()
    assert excinfo.value.name == 'dill'


def test_missing_ju_rjsf_submodule_raises(monkeypatch):
    """ju present but ju.rjsf absent (too-old ju) is a broken install, not 'absent'."""
    _require_ju()
    monkeypatch.setitem(sys.modules, 'ju.rjsf', None)
    monkeypatch.delattr(sys.modules['ju'], 'rjsf', raising=False)
    with pytest.raises(ModuleNotFoundError) as excinfo:
        _import_func_to_form_spec()
    assert excinfo.value.name == 'ju.rjsf'


def test_other_import_errors_propagate(monkeypatch):
    """An ImportError that is not 'ju is absent' (e.g. a renamed symbol) raises."""
    _require_ju()
    fake = type(sys)('ju.rjsf')  # module without func_to_form_spec
    monkeypatch.setitem(sys.modules, 'ju.rjsf', fake)
    monkeypatch.setattr(sys.modules['ju'], 'rjsf', fake, raising=False)
    with pytest.raises(ImportError, match='func_to_form_spec'):
        _import_func_to_form_spec()


def _variadic(a: int, b: str = "x", *args, c: float = 1.0, **kwargs):
    """Has *args and **kwargs."""


@pytest.mark.parametrize("with_ju", [True, False])
def test_variadic_params_are_skipped(with_ju, monkeypatch):
    """*args/**kwargs must not become required string fields (#9)."""
    import uf.specs

    if with_ju:
        _require_ju()
    else:
        monkeypatch.setattr(uf.specs, '_import_func_to_form_spec', lambda: None)
    schema = FunctionSpecStore([_variadic])['_variadic']['schema']
    assert set(schema['properties']) == {'a', 'b', 'c'}
    assert schema['required'] == ['a']


def test_param_to_prop_type_reaches_ju():
    """The param_to_prop_type option has an effect (#9)."""
    _require_ju()
    store = FunctionSpecStore([sample_add], param_to_prop_type=lambda param: 'string')
    props = store['sample_add']['schema']['properties']
    assert props['x']['type'] == 'string' and props['y']['type'] == 'string'
    # default (None) leaves ju's own mapping alone
    props = FunctionSpecStore([sample_add])['sample_add']['schema']['properties']
    assert props['x']['type'] == 'integer'


def test_variadic_names_follow_ju_and_uiSchema_is_cleaned():
    """Builtins whose ju name differs from inspect's, and a leading *args (#9)."""
    _require_ju()

    def f(*args, x: int, **kw): ...

    spec = FunctionSpecStore([f, print])
    ui = spec['f']['uiSchema']
    assert 'args' not in ui and ui.get('x', {}).get('ui:autofocus')
    assert 'value' not in spec['print']['schema']['properties']
    assert 'value' not in spec['print']['schema']['required']
