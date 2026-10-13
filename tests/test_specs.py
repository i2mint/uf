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
    """A leading *args: its uiSchema entry goes, the autofocus moves on (#9)."""
    _require_ju()

    def f(*args, x: int, **kw): ...

    spec = FunctionSpecStore([f])
    ui = spec['f']['uiSchema']
    assert 'args' not in ui and ui.get('x', {}).get('ui:autofocus')


# --- edge cases (i2mint/uf#10) ----------------------------------------------
from uf.specs import PositionalOnlyParamWarning, UnintrospectableCallableError


def _positional_only(a=1, b=2, /, c=3):
    """a and b can't be passed by keyword."""


def _required_positional_only_and_kw(a, /, b: int):
    """Needs a positionally."""


def _with_ju_or_not(with_ju, monkeypatch):
    import uf.specs

    if with_ju:
        _require_ju()
    else:
        monkeypatch.setattr(uf.specs, '_import_func_to_form_spec', lambda: None)


@pytest.mark.parametrize('with_ju', [True, False])
@pytest.mark.parametrize('func', [max, min, zip, getattr])
def test_unintrospectable_callable_raises_clear_error(with_ju, func, monkeypatch):
    _with_ju_or_not(with_ju, monkeypatch)
    store = FunctionSpecStore([func])  # lazy: building the store is fine
    assert func.__name__ in store  # membership does not generate the spec
    with pytest.raises(UnintrospectableCallableError, match=func.__name__):
        store[func.__name__]


@pytest.mark.parametrize('with_ju', [True, False])
def test_positional_only_params_are_dropped(with_ju, monkeypatch):
    _with_ju_or_not(with_ju, monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter('error', PositionalOnlyParamWarning)  # all defaulted
        schema = FunctionSpecStore([_positional_only])['_positional_only']['schema']
    assert set(schema['properties']) == {'c'}
    assert schema.get('required', []) == []


@pytest.mark.parametrize('with_ju', [True, False])
def test_required_positional_only_param_warns(with_ju, monkeypatch):
    _with_ju_or_not(with_ju, monkeypatch)
    store = FunctionSpecStore([_required_positional_only_and_kw])
    with pytest.warns(PositionalOnlyParamWarning, match=r"\['a'\]"):
        schema = store['_required_positional_only_and_kw']['schema']
    assert set(schema['properties']) == {'b'}
    assert schema['required'] == ['b']


def test_positional_only_uiSchema_autofocus_moves_on():
    _require_ju()
    ui = FunctionSpecStore([_positional_only])['_positional_only']['uiSchema']
    assert 'a' not in ui and ui.get('c', {}).get('ui:autofocus')


def test_nest_under_field_is_unwrapped():
    _require_ju()
    flat = FunctionSpecStore([sample_add])['sample_add']
    nested = FunctionSpecStore([sample_add], rjsf_config={'nest_under_field': 'rjsf'})
    assert nested['sample_add']['schema'] == flat['schema'] != {}
    assert nested['sample_add']['uiSchema'] == flat['uiSchema']


@pytest.mark.parametrize(
    'options, named',
    [
        ({'rjsf_config': {'liveValidate': True}}, 'rjsf_config'),
        ({'ui_schema_factory': lambda f: {}}, 'ui_schema_factory'),
        ({'param_to_prop_type': lambda p: 'string'}, 'param_to_prop_type'),
    ],
)
def test_fallback_warns_about_ignored_options(options, named, monkeypatch):
    _with_ju_or_not(False, monkeypatch)
    store = FunctionSpecStore([sample_add], **options)
    with pytest.warns(JuNotInstalledWarning, match=named):
        store['sample_add']


def test_fallback_without_options_has_no_ignored_options_warning(monkeypatch):
    _with_ju_or_not(False, monkeypatch)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        FunctionSpecStore([sample_add])['sample_add']
    assert not any('ignored' in str(w.message) for w in record)


@pytest.mark.parametrize('with_ju', [True, False])
def test_warnings_are_attributed_to_the_caller(with_ju, monkeypatch):
    _with_ju_or_not(with_ju, monkeypatch)
    store = FunctionSpecStore(
        [_required_positional_only_and_kw], param_to_prop_type=lambda p: 'string'
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        store['_required_positional_only_and_kw']
    assert record and all(w.filename == __file__ for w in record)


# --- optional-dependency guards (i2mint/uf#7) -------------------------------
from uf.optional import import_optional


class _AbsentFinder(importlib.abc.MetaPathFinder):
    """Make ``import <name>`` fail as if the package were not installed."""

    def __init__(self, name):
        self.name = name

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name:
            raise ModuleNotFoundError(f'No module named {self.name!r}', name=self.name)


class _BrokenPackageFinder(importlib.abc.MetaPathFinder):
    """Make ``import <name>`` fail as an installed-but-broken package would: with a
    ModuleNotFoundError naming a missing dependency *of* it."""

    def __init__(self, name):
        self.name = name

    def find_spec(self, fullname, path, target=None):
        if fullname == self.name:
            raise ModuleNotFoundError(
                f'No module named {self.name + "_dep"!r}', name=self.name + '_dep'
            )


def _break(monkeypatch, name):
    monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, 'meta_path', [_BrokenPackageFinder(name), *sys.meta_path])


def test_import_optional_absent_and_broken(monkeypatch):
    assert import_optional('_uf_no_such_package_') is None
    _break(monkeypatch, 'fakepkg')
    with pytest.raises(ModuleNotFoundError, match='fakepkg_dep'):
        import_optional('fakepkg')


@pytest.mark.parametrize('package', ['pandas', 'pydantic'])
def test_broken_optional_package_is_not_silently_ignored(package, monkeypatch):
    pytest.importorskip(package)
    from uf.pydantic_support import pydantic_error_to_user_friendly, is_pydantic_model
    from uf.renderers import DataFrameRenderer

    probe = {
        'pandas': lambda: DataFrameRenderer().can_render(object()),
        'pydantic': lambda: is_pydantic_model(object()),
    }[package]
    assert probe() is False  # healthy install: plain answer
    _break(monkeypatch, package)
    with pytest.raises(ModuleNotFoundError, match=f'{package}_dep'):
        probe()
    if package == 'pydantic':
        from uf.pydantic_support import extract_pydantic_params, function_uses_pydantic

        def f(x: int): ...

        for call in (
            lambda: pydantic_error_to_user_friendly(ValueError('x')),
            lambda: function_uses_pydantic(f),
            lambda: extract_pydantic_params(f),
        ):
            with pytest.raises(ModuleNotFoundError, match='pydantic_dep'):
                call()


@pytest.mark.parametrize('package', ['pandas', 'pydantic'])
def test_absent_optional_package_degrades(package, monkeypatch):
    from uf.pydantic_support import pydantic_error_to_user_friendly, is_pydantic_model
    from uf.renderers import DataFrameRenderer

    monkeypatch.delitem(sys.modules, package, raising=False)
    monkeypatch.setattr(sys, 'meta_path', [_AbsentFinder(package), *sys.meta_path])
    assert DataFrameRenderer().can_render(object()) is False
    assert is_pydantic_model(object()) is False
    assert pydantic_error_to_user_friendly(ValueError('x')) == {'error': 'x'}


def test_missing_qh_gets_the_install_hint_but_broken_qh_propagates(monkeypatch):
    from uf import mk_rjsf_app

    _break(monkeypatch, 'qh')
    with pytest.raises(ModuleNotFoundError, match='qh_dep'):
        mk_rjsf_app([sample_add])
    monkeypatch.setattr(sys, 'meta_path', [_AbsentFinder('qh'), *sys.meta_path[1:]])
    with pytest.raises(ImportError, match='pip install qh'):
        mk_rjsf_app([sample_add])


def test_import_optional_treats_a_namespace_directory_as_absent(tmp_path, monkeypatch):
    # a stray directory named like the optional package (no __init__.py)
    (tmp_path / '_uf_stray_dir_pkg_').mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, '_uf_stray_dir_pkg_', raising=False)
    assert import_optional('_uf_stray_dir_pkg_') is None
    # ... while a real package of that name is imported
    (tmp_path / '_uf_real_pkg_').mkdir()
    (tmp_path / '_uf_real_pkg_' / '__init__.py').write_text('X = 1\n')
    monkeypatch.delitem(sys.modules, '_uf_real_pkg_', raising=False)
    assert import_optional('_uf_real_pkg_').X == 1


def test_import_optional_dotted_name_under_a_namespace_directory(tmp_path, monkeypatch):
    (tmp_path / '_uf_stray_dir2_').mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, '_uf_stray_dir2_', raising=False)
    assert import_optional('_uf_stray_dir2_.sub', absent=('_uf_stray_dir2_',)) is None
    monkeypatch.delitem(sys.modules, '_uf_stray_dir2_', raising=False)
    assert import_optional('_uf_stray_dir2_.sub') is None


def test_import_optional_keeps_a_stub_package(monkeypatch):
    import types

    stub = types.ModuleType('_uf_stub_pkg_')
    stub.__path__ = []  # looks like a package, has no spec
    monkeypatch.setitem(sys.modules, '_uf_stub_pkg_', stub)
    assert import_optional('_uf_stub_pkg_') is stub
