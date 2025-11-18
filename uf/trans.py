"""Input transformation integration for uf.

Provides a registry for custom type transformations that bridges between
RJSF form data and qh's input transformation system.
"""

from typing import Callable, Optional, Any, Type
from collections.abc import Mapping
from uf.rjsf_config import RjsfFieldConfig


class InputTransformRegistry:
    """Registry for custom type transformations.

    Integrates with qh's type registry and extends it for UI needs,
    allowing custom types to be properly handled in both the form
    interface and the HTTP service layer.

    Example:
        >>> from datetime import datetime
        >>> registry = InputTransformRegistry()
        >>>
        >>> # Register a custom type
        >>> registry.register_type(
        ...     datetime,
        ...     to_json=lambda dt: dt.isoformat(),
        ...     from_json=lambda s: datetime.fromisoformat(s),
        ...     ui_widget='datetime'
        ... )
    """

    def __init__(self):
        """Initialize the transformation registry."""
        self._type_handlers: dict[Type, dict] = {}

    def register_type(
        self,
        py_type: Type,
        *,
        to_json: Optional[Callable[[Any], Any]] = None,
        from_json: Optional[Callable[[Any], Any]] = None,
        ui_widget: Optional[str] = None,
        ui_config: Optional[RjsfFieldConfig] = None,
        json_schema_type: Optional[str] = None,
        json_schema_format: Optional[str] = None,
    ) -> None:
        """Register a type with both qh and UI configuration.

        Args:
            py_type: Python type to register
            to_json: Function to convert Python type to JSON-serializable
            from_json: Function to convert JSON to Python type
            ui_widget: RJSF widget to use for this type
            ui_config: Full RjsfFieldConfig for this type
            json_schema_type: JSON Schema type (e.g., 'string', 'number')
            json_schema_format: JSON Schema format (e.g., 'date-time', 'email')

        Example:
            >>> from pathlib import Path
            >>> registry.register_type(
            ...     Path,
            ...     to_json=str,
            ...     from_json=Path,
            ...     ui_widget='text',
            ...     json_schema_type='string'
            ... )
        """
        handler = {
            'to_json': to_json or (lambda x: x),
            'from_json': from_json or (lambda x: x),
        }

        # Build UI config
        if ui_config:
            handler['ui_config'] = ui_config
        else:
            # Build from individual params
            config = RjsfFieldConfig(widget=ui_widget)
            if json_schema_format:
                config.format = json_schema_format
            handler['ui_config'] = config

        if json_schema_type:
            handler['json_schema_type'] = json_schema_type
        if json_schema_format:
            handler['json_schema_format'] = json_schema_format

        self._type_handlers[py_type] = handler

    def get_handler(self, py_type: Type) -> Optional[dict]:
        """Get handler for a type.

        Args:
            py_type: Python type to look up

        Returns:
            Handler dict or None if not registered
        """
        return self._type_handlers.get(py_type)

    def get_ui_config(self, py_type: Type) -> Optional[RjsfFieldConfig]:
        """Get UI configuration for a type.

        Args:
            py_type: Python type to look up

        Returns:
            RjsfFieldConfig or None if not registered
        """
        handler = self.get_handler(py_type)
        if handler:
            return handler.get('ui_config')
        return None

    def to_json(self, value: Any, py_type: Optional[Type] = None) -> Any:
        """Transform a Python value to JSON-serializable form.

        Args:
            value: Value to transform
            py_type: Optional type hint (uses type(value) if not provided)

        Returns:
            JSON-serializable value
        """
        if value is None:
            return None

        target_type = py_type or type(value)
        handler = self.get_handler(target_type)

        if handler and handler['to_json']:
            return handler['to_json'](value)

        return value

    def from_json(self, value: Any, py_type: Type) -> Any:
        """Transform a JSON value to Python type.

        Args:
            value: JSON value to transform
            py_type: Target Python type

        Returns:
            Transformed value
        """
        if value is None:
            return None

        handler = self.get_handler(py_type)

        if handler and handler['from_json']:
            return handler['from_json'](value)

        return value

    def mk_input_trans_for_funcs(
        self,
        funcs: list[Callable],
    ) -> Callable:
        """Create input transformation compatible with qh.

        This creates a transformation function that can be passed to
        qh.mk_app as the input_trans parameter.

        Args:
            funcs: List of functions to create transformation for

        Returns:
            Transformation function for qh

        Example:
            >>> from uf import mk_rjsf_app
            >>> registry = InputTransformRegistry()
            >>> # ... register types ...
            >>> input_trans = registry.mk_input_trans_for_funcs([my_func])
            >>> app = mk_rjsf_app([my_func], input_trans=input_trans)
        """
        import inspect

        # Build mapping of func_name -> param_name -> type
        func_type_map = {}
        for func in funcs:
            sig = inspect.signature(func)
            param_types = {}
            for param_name, param in sig.parameters.items():
                if param.annotation != inspect.Parameter.empty:
                    param_types[param_name] = param.annotation
            func_type_map[func.__name__] = param_types

        def input_trans(func_name: str, kwargs: dict) -> dict:
            """Transform input kwargs based on registered types."""
            if func_name not in func_type_map:
                return kwargs

            param_types = func_type_map[func_name]
            transformed = {}

            for param_name, value in kwargs.items():
                if param_name in param_types:
                    py_type = param_types[param_name]
                    transformed[param_name] = self.from_json(value, py_type)
                else:
                    transformed[param_name] = value

            return transformed

        return input_trans

    def mk_output_trans(self) -> Callable:
        """Create output transformation for qh.

        Returns:
            Transformation function for qh output

        Example:
            >>> output_trans = registry.mk_output_trans()
            >>> app = mk_rjsf_app([my_func], output_trans=output_trans)
        """

        def output_trans(result: Any) -> Any:
            """Transform output to JSON-serializable form."""
            if result is None:
                return None

            # Try to transform using registered handlers
            result_type = type(result)
            handler = self.get_handler(result_type)

            if handler and handler['to_json']:
                return handler['to_json'](result)

            # Handle common collection types
            if isinstance(result, list):
                return [output_trans(item) for item in result]
            elif isinstance(result, dict):
                return {k: output_trans(v) for k, v in result.items()}
            elif isinstance(result, tuple):
                return [output_trans(item) for item in result]

            return result

        return output_trans

    def get_all_registered_types(self) -> list[Type]:
        """Get list of all registered types.

        Returns:
            List of registered Python types
        """
        return list(self._type_handlers.keys())


# Global registry instance for convenience
_global_registry = InputTransformRegistry()


def register_type(*args, **kwargs):
    """Register a type in the global registry.

    This is a convenience function that uses the global registry.
    See InputTransformRegistry.register_type for full documentation.
    """
    return _global_registry.register_type(*args, **kwargs)


def get_global_registry() -> InputTransformRegistry:
    """Get the global transformation registry.

    Returns:
        The global InputTransformRegistry instance
    """
    return _global_registry


# Register common custom types
def register_common_types():
    """Register commonly-used Python types.

    This includes:
    - datetime.datetime
    - datetime.date
    - datetime.time
    - pathlib.Path
    - uuid.UUID
    - decimal.Decimal
    """
    from datetime import datetime, date, time
    from pathlib import Path
    from uuid import UUID
    from decimal import Decimal

    # datetime types
    _global_registry.register_type(
        datetime,
        to_json=lambda dt: dt.isoformat(),
        from_json=lambda s: datetime.fromisoformat(s) if isinstance(s, str) else s,
        ui_widget='datetime',
        json_schema_type='string',
        json_schema_format='date-time',
    )

    _global_registry.register_type(
        date,
        to_json=lambda d: d.isoformat(),
        from_json=lambda s: date.fromisoformat(s) if isinstance(s, str) else s,
        ui_widget='date',
        json_schema_type='string',
        json_schema_format='date',
    )

    _global_registry.register_type(
        time,
        to_json=lambda t: t.isoformat(),
        from_json=lambda s: time.fromisoformat(s) if isinstance(s, str) else s,
        ui_widget='time',
        json_schema_type='string',
        json_schema_format='time',
    )

    # Path
    _global_registry.register_type(
        Path,
        to_json=str,
        from_json=Path,
        json_schema_type='string',
    )

    # UUID
    _global_registry.register_type(
        UUID,
        to_json=str,
        from_json=lambda s: UUID(s) if isinstance(s, str) else s,
        json_schema_type='string',
        json_schema_format='uuid',
    )

    # Decimal
    _global_registry.register_type(
        Decimal,
        to_json=float,
        from_json=Decimal,
        json_schema_type='number',
    )


# Auto-register common types on import
try:
    register_common_types()
except ImportError:
    # Some types might not be available, that's okay
    pass
