"""Function specification management for uf.

Provides a Mapping-based interface to function specifications, including RJSF
form specs and OpenAPI schemas.
"""

from typing import Callable, Iterable, Optional, Any
from collections.abc import Mapping
from functools import cached_property


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

        try:
            # Import ju.rjsf for form spec generation
            from ju.rjsf import func_to_form_spec

            # Generate form spec
            form_spec = func_to_form_spec(
                func,
                **self._rjsf_config
            )

            # Apply custom UI schema factory if provided
            if self._ui_schema_factory:
                ui_schema = self._ui_schema_factory(func)
                if 'uiSchema' in form_spec:
                    form_spec['uiSchema'].update(ui_schema)
                else:
                    form_spec['uiSchema'] = ui_schema

            return {
                'schema': form_spec.get('schema', {}),
                'uiSchema': form_spec.get('uiSchema', {}),
                'func': func,
                'name': func_name,
                'description': func.__doc__ or f"Execute {func_name}",
            }

        except ImportError:
            # Fallback to basic spec if ju.rjsf not available
            return self._generate_basic_spec(func_name, func)

    def _generate_basic_spec(self, func_name: str, func: Callable) -> dict:
        """Generate a basic specification without ju.rjsf.

        This is a fallback for when ju.rjsf is not available. It creates
        a minimal JSON Schema from function signature.

        Args:
            func_name: Name of the function
            func: The function object

        Returns:
            Basic specification dictionary
        """
        import inspect

        sig = inspect.signature(func)
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
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
            'schema': {
                'type': 'object',
                'properties': properties,
                'required': required,
                'title': func_name,
                'description': func.__doc__ or '',
            },
            'uiSchema': {},
            'func': func,
            'name': func_name,
            'description': func.__doc__ or f"Execute {func_name}",
        }

    @cached_property
    def function_list(self) -> list[dict]:
        """Get list of all functions with basic metadata.

        Returns:
            List of dictionaries with function name and description
        """
        return [
            {
                'name': name,
                'description': func.__doc__ or f"Execute {name}",
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
