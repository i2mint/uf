"""Function specification management for uf.

Provides a Mapping-based interface to function specifications, including RJSF
form specs and OpenAPI schemas.
"""

import inspect
import os
import sys
import warnings
import functools
from typing import Callable, Iterable, Optional, Any, Union
from collections.abc import Mapping
from functools import cached_property

from uf.optional import import_optional

# Only the absence of top-level ``ju`` means "not installed" (fall back). A missing
# ``ju.rjsf`` (too-old ju, stray namespace dir) or a missing dependency *of* ju is
# a broken install: let it raise.
_JU_MODULE_NAMES = ("ju",)


_VARIADIC_KINDS = (
    inspect.Parameter.VAR_POSITIONAL,  # *args
    inspect.Parameter.VAR_KEYWORD,  # **kwargs
)
# uf calls functions with keyword arguments only, so a form can't fill these
_NON_KEYWORD_KINDS = _VARIADIC_KINDS + (inspect.Parameter.POSITIONAL_ONLY,)


class UnintrospectableCallableError(ValueError):
    """A callable has no signature to build a form from (e.g. ``max``, ``getattr``)."""


class UnservableCallableError(ValueError):
    """A callable can never be called from a uf form (e.g. it has a required
    positional-only parameter)."""


class FormSpecError(ValueError):
    """The form spec of a function could not be generated (the cause is chained)."""


def _check_introspectable(func: Callable) -> None:
    """Raise ``UnintrospectableCallableError`` if ``func`` has no usable signature.

    >>> _check_introspectable(len)
    >>> _check_introspectable(max)
    Traceback (most recent call last):
      ...
    uf.specs.UnintrospectableCallableError: 'max' has no introspectable signature; wrap it in a function with an explicit signature to serve it
    """
    try:
        inspect.signature(func)
    except (TypeError, ValueError) as e:
        name = getattr(func, "__name__", repr(func))
        raise UnintrospectableCallableError(
            f"{name!r} has no introspectable signature; wrap it in a function "
            "with an explicit signature to serve it"
        ) from e


def _check_servable(func: Callable, name: str) -> None:
    """Raise if ``func`` can never be called from a form: no signature, or a required
    positional-only parameter (uf calls functions with keyword arguments only, so
    every submission would fail).

    >>> _check_servable(len, 'len')
    Traceback (most recent call last):
      ...
    uf.specs.UnservableCallableError: 'len' takes required positional-only parameter(s) ['obj']; uf calls functions with keyword arguments only, so its form could never be submitted successfully
    >>> _check_servable(lambda a, /, b: None, 'f')
    Traceback (most recent call last):
      ...
    uf.specs.UnservableCallableError: 'f' takes required positional-only parameter(s) ['a']; uf calls functions with keyword arguments only, so its form could never be submitted successfully
    >>> _check_servable(lambda a=1, /, b=2: None, 'f')
    """
    _check_introspectable(func)
    params = inspect.signature(func).parameters.values()
    required_po = [
        p.name
        for p in params
        if p.kind is inspect.Parameter.POSITIONAL_ONLY
        and p.default is inspect.Parameter.empty
    ]
    if required_po:
        raise UnservableCallableError(
            f"{name!r} takes required positional-only parameter(s) {required_po}; "
            "uf calls functions with keyword arguments only, so its form could never "
            "be submitted successfully"
        )


def name_of_func(func: Callable) -> str:
    """The name a function is served under: its ``__name__``; failing that, for a
    ``functools.partial``, that of the function it wraps; for a callable instance, its
    class name.

    >>> import functools
    >>> name_of_func(len)
    'len'
    >>> name_of_func(functools.partial(int, base=2))
    'int'
    >>> class Doubler:
    ...     def __call__(self, x): return 2 * x
    >>> name_of_func(Doubler())
    'Doubler'
    """
    while not hasattr(func, "__name__") and isinstance(func, functools.partial):
        func = func.func
    return getattr(func, "__name__", None) or type(func).__name__


def doc_of_func(func: Callable) -> Optional[str]:
    """The docstring of ``func``; for a ``functools.partial`` (whose own ``__doc__`` is
    the generic one of ``partial``), that of the function it wraps.

    >>> import functools
    >>> def f(x, y):
    ...     "Add."
    >>> doc_of_func(functools.partial(f, y=1))
    'Add.'
    """
    while isinstance(func, functools.partial):
        func = func.func
    return func.__doc__


def name_funcs(funcs) -> dict:
    """``{name: callable}`` from a mapping of explicit names, or an iterable of
    callables (named by ``name_of_func``). Two different callables sharing a name
    raise.

    >>> import functools
    >>> name_funcs([len, functools.partial(int, base=2)])  # doctest: +ELLIPSIS
    {'len': <built-in function len>, 'int': functools.partial(<class 'int'>, base=2)}
    >>> name_funcs({'size': len})
    {'size': <built-in function len>}
    """
    if isinstance(funcs, Mapping):
        return dict(funcs)
    named = {}
    for f in funcs:
        name = name_of_func(f)
        if name in named and named[name] != f:
            raise ValueError(
                f"two different callables are both named {name!r}; pass a mapping "
                "{name: callable} to name them explicitly"
            )
        named[name] = f
    return named


def _unfillable_names(func: Callable) -> set:
    """Names of ``func``'s parameters a keyword-only caller can't fill (``*args``,
    ``**kwargs``, positional-only).

    Both ``i2.Sig`` and ``inspect`` are asked: they name the variadic parameter of some
    callables differently (``print``: ``*value`` vs ``*args``), and ju may use either.

    >>> sorted(_unfillable_names(lambda a, /, b, *args, **kw: None))
    ['a', 'args', 'kw']
    """
    names = set()
    try:
        from i2 import Sig

        names |= _non_keyword_names(Sig(func).parameters)
    except Exception:  # no i2, or no introspectable signature
        pass
    try:
        names |= _non_keyword_names(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        pass
    return names


def _non_keyword_names(params) -> set:
    return {n for n, p in params.items() if p.kind in _NON_KEYWORD_KINDS}


def _drop_unfillable_fields(func: Callable, form_spec: dict) -> None:
    """Remove ``*args``/``**kwargs`` and positional-only parameters from a form spec's
    schema and uiSchema (in place).

    A form can't sensibly ask for them (``*args``/``**kwargs`` would show up as
    required string fields), and uf calls functions with keyword arguments only.
    """
    variadic = _unfillable_names(func)
    schema = form_spec.get("schema", {})
    for name in variadic:
        schema.get("properties", {}).pop(name, None)
    if "required" in schema:
        schema["required"] = [n for n in schema["required"] if n not in variadic]
    ui_schema = form_spec.get("uiSchema", {})
    dropped = [ui_schema.pop(name) for name in variadic if name in ui_schema]
    # ju puts the autofocus on the first parameter: hand it to the first one left
    if any(isinstance(d, dict) and d.get("ui:autofocus") for d in dropped):
        for name in schema.get("properties", {}):
            ui_schema.setdefault(name, {})["ui:autofocus"] = True
            break


class JuNotInstalledWarning(UserWarning):
    """Emitted when ``ju`` is absent and uf falls back to basic form specs."""


_PACKAGE_PREFIX = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _warn_at_caller(message: str, category: type[Warning]) -> None:
    """Warn, attributed to the first frame outside the ``uf`` package.

    A hardcoded ``stacklevel`` is right for one call path only: the same warning
    reached through ``mk_rjsf_app`` or ``UfApp`` would be attributed to uf's own
    ``base.py``. Walking up to the first frame that is not uf's points at the
    user's line however the store was built.

    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as w:
    ...     warnings.simplefilter("always")
    ...     _warn_at_caller("hi", UserWarning)
    >>> w[0].filename.startswith("<doctest")  # no uf frame above: the doctest itself
    True
    """
    frame = sys._getframe(1)
    stacklevel = 2  # 1 would be this function; 2 is its caller
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_PREFIX):
        frame = frame.f_back
        stacklevel += 1
    warnings.warn(message, category, stacklevel=stacklevel)


def _import_func_to_form_spec() -> Optional[Callable]:
    """Return ``ju.rjsf.func_to_form_spec``, or ``None`` if ``ju`` is not installed.
    (The warning is attributed to the first caller outside the ``uf`` package.)

    Only the absence of ``ju`` itself is tolerated (with a visible warning).
    Any other import failure -- a missing dependency of ``ju``, a renamed
    symbol -- is a broken environment and propagates, so it fails loudly
    instead of silently degrading every generated form.

    >>> _import_func_to_form_spec() is None or callable(_import_func_to_form_spec())
    True
    """
    rjsf = import_optional("ju.rjsf", absent=_JU_MODULE_NAMES)
    if rjsf is None:
        _warn_at_caller(
            "'ju' is not installed; uf is falling back to basic form "
            "specs (no ju/RJSF features). Install it with `pip install ju`.",
            JuNotInstalledWarning,
        )
        return None
    try:
        return rjsf.func_to_form_spec
    except AttributeError as e:  # a renamed symbol: a broken ju, not an absent one
        raise ImportError(
            "cannot import name 'func_to_form_spec' from 'ju.rjsf'"
        ) from e


class FunctionSpecStore(Mapping):
    """A mapping from function names to their RJSF specifications.

    Lazily generates and caches form specs for each function.

    Args:
        funcs: Iterable of callables to generate specs for (named by ``name_of_func``:
            ``__name__``, the wrapped function's for a ``functools.partial``, the class
            name for a callable instance), or a mapping ``{name: callable}`` to name
            them explicitly
        rjsf_config: Optional configuration dict for RJSF generation
        ui_schema_factory: Optional callable to customize UI schema generation
        param_to_prop_type: Optional callable to map parameters to property types
        validate: If true, every function is checked and its spec generated at
            construction, so a callable uf can't build a form for fails there, naming
            it, rather than on the first request (default false: checked on first
            access of its spec; ``mk_rjsf_app`` validates).

    Example:
        >>> def add(x: int, y: int) -> int:
        ...     '''Add two numbers'''
        ...     return x + y
        >>> specs = FunctionSpecStore([add])
        >>> 'add' in specs
        True
        >>> spec = specs['add']
        >>> 'schema' in spec
        True
    """

    def __init__(
        self,
        funcs: Union[Iterable[Callable], Mapping[str, Callable]],
        *,
        rjsf_config: Optional[dict] = None,
        ui_schema_factory: Optional[Callable] = None,
        param_to_prop_type: Optional[Callable] = None,
        validate: bool = False,
    ):
        self._funcs = name_funcs(funcs)
        if validate:  # (before anything else can fail, so the function is named)
            for name, func in self._funcs.items():
                _check_servable(func, name)
        self._rjsf_config = rjsf_config or {}
        self._ui_schema_factory = ui_schema_factory
        self._param_to_prop_type = param_to_prop_type
        self._spec_cache = {}
        # Resolved once, here: a broken ju fails the build (with its traceback), not
        # every request, and the absent-ju warning points at the code building the store.
        self._func_to_form_spec = _import_func_to_form_spec()
        if self._func_to_form_spec is None:
            self._warn_ignored_options()
        if validate:
            for name in self._funcs:
                try:
                    self[name]  # (generates and caches the spec)
                except (UnservableCallableError, UnintrospectableCallableError):
                    raise
                except Exception as e:
                    raise FormSpecError(
                        f"could not generate the form spec of {name!r}: {e!r}"
                    ) from e

    def __getitem__(self, func_name: str) -> dict:
        """Get RJSF spec for a function.

        Args:
            func_name: Name of the function

        Returns:
            Dictionary containing the RJSF specification with keys:
            - 'schema': JSON Schema for the function inputs
            - 'uiSchema': UI Schema for rendering hints
            - 'func': The original function object

        Raises:
            KeyError: If function name not found
        """
        if func_name not in self._funcs:
            raise KeyError(f"Function '{func_name}' not found")

        if func_name not in self._spec_cache:
            self._spec_cache[func_name] = self._generate_spec(func_name)

        return self._spec_cache[func_name]

    def __iter__(self):
        """Iterate over function names."""
        return iter(self._funcs)

    def __contains__(self, func_name) -> bool:
        """Whether a function is served (does not generate its spec)."""
        return func_name in self._funcs

    def __len__(self):
        """Return number of functions."""
        return len(self._funcs)

    def _generate_spec(self, func_name: str) -> dict:
        """Generate RJSF specification for a function.

        Args:
            func_name: Name of the function to generate spec for

        Returns:
            Dictionary with schema, uiSchema, and function reference
        """
        func = self._funcs[func_name]
        _check_servable(func, func_name)

        func_to_form_spec = self._func_to_form_spec
        if func_to_form_spec is None:
            return self._generate_basic_spec(func_name, func)

        # ju nests the spec under a "rjsf" field by default; uf wants it flat
        rjsf_config = {"nest_under_field": None}
        if self._param_to_prop_type is not None:
            rjsf_config["param_to_prop_type"] = self._param_to_prop_type
        rjsf_config.update(self._rjsf_config)  # explicit rjsf_config wins
        form_spec = func_to_form_spec(func, **rjsf_config)
        nest_under_field = rjsf_config.get("nest_under_field")
        if nest_under_field:  # the caller asked ju to nest it; uf reads it flat
            form_spec = form_spec[nest_under_field]
        _drop_unfillable_fields(func, form_spec)

        # Apply custom UI schema factory if provided
        if self._ui_schema_factory:
            ui_schema = self._ui_schema_factory(func)
            if "uiSchema" in form_spec:
                form_spec["uiSchema"].update(ui_schema)
            else:
                form_spec["uiSchema"] = ui_schema

        return {
            "schema": form_spec.get("schema", {}),
            "uiSchema": form_spec.get("uiSchema", {}),
            "func": func,
            "name": func_name,
            "description": doc_of_func(func) or f"Execute {func_name}",
        }

    def _warn_ignored_options(self) -> None:
        """Warn about the options the basic fallback has no way to honour."""
        ignored = [
            name
            for name, value in (
                ("rjsf_config", self._rjsf_config),
                ("ui_schema_factory", self._ui_schema_factory),
                ("param_to_prop_type", self._param_to_prop_type),
            )
            if value
        ]
        if ignored:
            _warn_at_caller(
                f"{ignored} {'is' if len(ignored) == 1 else 'are'} ignored because "
                "ju is not installed (basic form specs). Install it with "
                "`pip install ju`.",
                JuNotInstalledWarning,
            )

    def _generate_basic_spec(self, func_name: str, func: Callable) -> dict:
        """Generate a basic specification without ju.rjsf.

        This is the fallback used when ``ju`` is not installed. It creates
        a minimal JSON Schema from function signature.

        Args:
            func_name: Name of the function
            func: The function object

        Returns:
            Basic specification dictionary
        """
        sig = inspect.signature(func)
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param.kind in _NON_KEYWORD_KINDS:
                continue
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

            # Basic type mapping
            param_type = "string"  # default
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"

            properties[param_name] = {"type": param_type}

        return {
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "title": func_name,
                "description": doc_of_func(func) or "",
            },
            "uiSchema": {},
            "func": func,
            "name": func_name,
            "description": doc_of_func(func) or f"Execute {func_name}",
        }

    @cached_property
    def function_list(self) -> list[dict]:
        """Get list of all functions with basic metadata.

        Returns:
            List of dictionaries with function name and description
        """
        return [
            {
                "name": name,
                "description": doc_of_func(func) or f"Execute {name}",
            }
            for name, func in self._funcs.items()
        ]

    def get_func(self, func_name: str) -> Callable:
        """Get the original function object by name.

        Args:
            func_name: Name of the function

        Returns:
            The function object

        Raises:
            KeyError: If function name not found
        """
        return self._funcs[func_name]
