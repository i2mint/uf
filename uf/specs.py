"""Function specification management for uf.

Provides a Mapping-based interface to function specifications, including RJSF
form specs and OpenAPI schemas.
"""

import inspect
import warnings
from typing import Callable, Iterable, Optional, Any
from collections.abc import Mapping
from functools import cached_property

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


class PositionalOnlyParamWarning(UserWarning):
    """A required positional-only parameter can't be filled by a keyword-only caller."""


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


def _unfillable_names(func: Callable) -> set:
    """Names of ``func``'s parameters a keyword-only caller can't fill (``*args``,
    ``**kwargs``, positional-only), as ju (``i2.Sig``) names them."""
    try:
        from i2 import Sig

        params = Sig(func).parameters
    except Exception:  # no i2, or no introspectable signature
        try:
            params = inspect.signature(func).parameters
        except (TypeError, ValueError):
            return set()
    return {n for n, p in params.items() if p.kind in _NON_KEYWORD_KINDS}


def _warn_required_positional_only(
    func: Callable, func_name: str, *, stacklevel: int
) -> None:
    """Warn if ``func`` has a required positional-only parameter (it is left out of the
    form, so calling the function with keyword arguments fails)."""
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return
    required = [
        p.name
        for p in params
        if p.kind is inspect.Parameter.POSITIONAL_ONLY
        and p.default is inspect.Parameter.empty
    ]
    if required:
        warnings.warn(
            f"{func_name!r} has required positional-only parameter(s) {required}, "
            "which uf can't pass (it calls functions with keyword arguments only); "
            "they are left out of the form.",
            PositionalOnlyParamWarning,
            stacklevel=stacklevel,
        )


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


def _import_func_to_form_spec() -> Optional[Callable]:
    """Return ``ju.rjsf.func_to_form_spec``, or ``None`` if ``ju`` is not installed.

    Only the absence of ``ju`` itself is tolerated (with a visible warning).
    Any other import failure -- a missing dependency of ``ju``, a renamed
    symbol -- is a broken environment and propagates, so it fails loudly
    instead of silently degrading every generated form.

    >>> _import_func_to_form_spec() is None or callable(_import_func_to_form_spec())
    True
    """
    try:
        from ju.rjsf import func_to_form_spec
    except ModuleNotFoundError as e:
        if e.name not in _JU_MODULE_NAMES:
            raise
        warnings.warn(
            f"{e.name!r} is not installed; uf is falling back to basic form "
            "specs (no ju/RJSF features). Install it with `pip install ju`.",
            JuNotInstalledWarning,
            stacklevel=4,  # caller -> __getitem__ -> _generate_spec -> here
        )
        return None
    return func_to_form_spec


class FunctionSpecStore(Mapping):
    """A mapping from function names to their RJSF specifications.

    Lazily generates and caches form specs for each function.

    Args:
        funcs: Iterable of callable functions to generate specs for
        rjsf_config: Optional configuration dict for RJSF generation
        ui_schema_factory: Optional callable to customize UI schema generation
        param_to_prop_type: Optional callable to map parameters to property types

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
        funcs: Iterable[Callable],
        *,
        rjsf_config: Optional[dict] = None,
        ui_schema_factory: Optional[Callable] = None,
        param_to_prop_type: Optional[Callable] = None,
    ):
        self._funcs = {f.__name__: f for f in funcs}
        self._rjsf_config = rjsf_config or {}
        self._ui_schema_factory = ui_schema_factory
        self._param_to_prop_type = param_to_prop_type
        self._spec_cache = {}

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
        _check_introspectable(func)

        func_to_form_spec = _import_func_to_form_spec()
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
        # caller -> __getitem__ -> _generate_spec -> warn helper
        _warn_required_positional_only(func, func_name, stacklevel=4)
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
            "description": func.__doc__ or f"Execute {func_name}",
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
            warnings.warn(
                f"{ignored} {'is' if len(ignored) == 1 else 'are'} ignored because "
                "ju is not installed (basic form specs). Install it with "
                "`pip install ju`.",
                JuNotInstalledWarning,
                stacklevel=5,  # caller -> __getitem__ -> _generate_spec -> basic -> here
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
        self._warn_ignored_options()
        # caller -> __getitem__ -> _generate_spec -> basic spec -> warn helper
        _warn_required_positional_only(func, func_name, stacklevel=5)
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
                "description": func.__doc__ or "",
            },
            "uiSchema": {},
            "func": func,
            "name": func_name,
            "description": func.__doc__ or f"Execute {func_name}",
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
                "description": func.__doc__ or f"Execute {name}",
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
