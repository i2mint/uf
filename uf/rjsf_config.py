"""RJSF customization layer for uf.

Provides configuration classes and builders for customizing RJSF form
generation beyond the defaults, including field widgets, UI options,
and validation rules.
"""

from typing import Optional, Callable, Any, Literal
from dataclasses import dataclass, field


@dataclass
class RjsfFieldConfig:
    """Configuration for individual form fields.

    This class allows fine-grained control over how form fields are
    rendered in the RJSF interface.

    Attributes:
        widget: Widget type (e.g., 'textarea', 'select', 'radio', 'date')
        ui_options: Additional UI options for the widget
        format: JSON Schema format (e.g., 'email', 'uri', 'date-time')
        enum: List of allowed values (for dropdowns)
        description: Field description/help text
        placeholder: Placeholder text for the input
        title: Custom title for the field
        disabled: Whether the field is disabled
        readonly: Whether the field is read-only
        hidden: Whether to hide the field
        default: Default value for the field

    Example:
        >>> email_config = RjsfFieldConfig(
        ...     widget='email',
        ...     format='email',
        ...     placeholder='user@example.com'
        ... )
    """

    widget: Optional[str] = None
    ui_options: dict = field(default_factory=dict)
    format: Optional[str] = None
    enum: Optional[list] = None
    description: Optional[str] = None
    placeholder: Optional[str] = None
    title: Optional[str] = None
    disabled: bool = False
    readonly: bool = False
    hidden: bool = False
    default: Optional[Any] = None

    def to_json_schema_patch(self) -> dict:
        """Convert to JSON Schema properties.

        Returns:
            Dictionary of JSON Schema properties to merge into schema
        """
        patch = {}

        if self.format:
            patch['format'] = self.format
        if self.enum:
            patch['enum'] = self.enum
        if self.description:
            patch['description'] = self.description
        if self.title:
            patch['title'] = self.title
        if self.default is not None:
            patch['default'] = self.default

        return patch

    def to_ui_schema_patch(self) -> dict:
        """Convert to RJSF UI Schema properties.

        Returns:
            Dictionary of UI Schema properties for this field
        """
        ui_patch = {}

        if self.widget:
            ui_patch['ui:widget'] = self.widget
        if self.placeholder:
            ui_patch['ui:placeholder'] = self.placeholder
        if self.disabled:
            ui_patch['ui:disabled'] = True
        if self.readonly:
            ui_patch['ui:readonly'] = True
        if self.hidden:
            ui_patch['ui:widget'] = 'hidden'

        if self.ui_options:
            ui_patch['ui:options'] = self.ui_options

        return ui_patch


class RjsfConfigBuilder:
    """Builder for RJSF configurations with sensible defaults.

    This class helps construct RJSF specifications by providing a
    fluent interface for configuring fields.

    Example:
        >>> builder = RjsfConfigBuilder()
        >>> builder.field('email', RjsfFieldConfig(format='email'))
        >>> builder.field('message', RjsfFieldConfig(widget='textarea'))
        >>> spec = builder.build(base_schema)
    """

    def __init__(self):
        """Initialize the config builder."""
        self._field_configs: dict[str, RjsfFieldConfig] = {}
        self._ui_order: Optional[list[str]] = None
        self._class_names: Optional[str] = None

    def field(self, param_name: str, config: RjsfFieldConfig) -> 'RjsfConfigBuilder':
        """Configure a specific field.

        Args:
            param_name: Name of the parameter/field
            config: Configuration for the field

        Returns:
            Self for method chaining
        """
        self._field_configs[param_name] = config
        return self

    def order(self, field_order: list[str]) -> 'RjsfConfigBuilder':
        """Set the order of fields in the form.

        Args:
            field_order: List of field names in desired order

        Returns:
            Self for method chaining
        """
        self._ui_order = field_order
        return self

    def class_names(self, class_names: str) -> 'RjsfConfigBuilder':
        """Set CSS class names for the form.

        Args:
            class_names: Space-separated CSS class names

        Returns:
            Self for method chaining
        """
        self._class_names = class_names
        return self

    def build(self, base_schema: dict, base_ui_schema: Optional[dict] = None) -> dict:
        """Build the final RJSF specification.

        Args:
            base_schema: Base JSON Schema to augment
            base_ui_schema: Optional base UI Schema to augment

        Returns:
            Dictionary with 'schema' and 'uiSchema' keys
        """
        schema = base_schema.copy()
        ui_schema = (base_ui_schema or {}).copy()

        # Apply field configurations
        for field_name, config in self._field_configs.items():
            # Update schema properties
            if 'properties' in schema and field_name in schema['properties']:
                schema_patch = config.to_json_schema_patch()
                schema['properties'][field_name].update(schema_patch)

            # Update UI schema
            ui_patch = config.to_ui_schema_patch()
            if ui_patch:
                ui_schema[field_name] = {**ui_schema.get(field_name, {}), **ui_patch}

        # Apply UI order
        if self._ui_order:
            ui_schema['ui:order'] = self._ui_order

        # Apply class names
        if self._class_names:
            ui_schema['ui:classNames'] = self._class_names

        return {
            'schema': schema,
            'uiSchema': ui_schema,
        }


# Predefined field configurations for common use cases
COMMON_FIELD_CONFIGS = {
    'email': RjsfFieldConfig(
        widget='email',
        format='email',
        placeholder='user@example.com',
    ),
    'password': RjsfFieldConfig(
        widget='password',
    ),
    'url': RjsfFieldConfig(
        format='uri',
        placeholder='https://example.com',
    ),
    'multiline_text': RjsfFieldConfig(
        widget='textarea',
        ui_options={'rows': 5},
    ),
    'long_text': RjsfFieldConfig(
        widget='textarea',
        ui_options={'rows': 10},
    ),
    'date': RjsfFieldConfig(
        widget='date',
        format='date',
    ),
    'datetime': RjsfFieldConfig(
        widget='datetime',
        format='date-time',
    ),
    'color': RjsfFieldConfig(
        widget='color',
    ),
    'range': RjsfFieldConfig(
        widget='range',
    ),
    'file': RjsfFieldConfig(
        widget='file',
    ),
}


def get_field_config(config_name: str) -> RjsfFieldConfig:
    """Get a predefined field configuration by name.

    Args:
        config_name: Name of the configuration (e.g., 'email', 'multiline_text')

    Returns:
        RjsfFieldConfig instance

    Raises:
        KeyError: If config_name not found

    Example:
        >>> email_config = get_field_config('email')
        >>> email_config.format
        'email'
    """
    if config_name not in COMMON_FIELD_CONFIGS:
        raise KeyError(
            f"Unknown config '{config_name}'. "
            f"Available: {list(COMMON_FIELD_CONFIGS.keys())}"
        )
    return COMMON_FIELD_CONFIGS[config_name]


def apply_field_configs(
    schema: dict,
    ui_schema: dict,
    field_configs: dict[str, RjsfFieldConfig],
) -> tuple[dict, dict]:
    """Apply field configurations to existing schemas.

    Args:
        schema: JSON Schema to modify
        ui_schema: UI Schema to modify
        field_configs: Mapping of field names to configurations

    Returns:
        Tuple of (modified_schema, modified_ui_schema)

    Example:
        >>> configs = {
        ...     'email': get_field_config('email'),
        ...     'bio': get_field_config('multiline_text'),
        ... }
        >>> schema, ui_schema = apply_field_configs(schema, ui_schema, configs)
    """
    builder = RjsfConfigBuilder()
    for field_name, config in field_configs.items():
        builder.field(field_name, config)

    result = builder.build(schema, ui_schema)
    return result['schema'], result['uiSchema']


class ConditionalFieldConfig:
    """Configuration for conditional field display.

    Allows fields to be shown/hidden based on the values of other fields.

    Example:
        >>> # Show 'other_reason' field only when reason is 'other'
        >>> config = ConditionalFieldConfig(
        ...     'other_reason',
        ...     condition={'reason': {'const': 'other'}}
        ... )
    """

    def __init__(
        self,
        field_name: str,
        *,
        condition: dict,
        then_schema: Optional[dict] = None,
        else_schema: Optional[dict] = None,
    ):
        """Initialize conditional field configuration.

        Args:
            field_name: Name of the field to make conditional
            condition: JSON Schema condition (if/then/else style)
            then_schema: Schema to apply when condition is true
            else_schema: Schema to apply when condition is false
        """
        self.field_name = field_name
        self.condition = condition
        self.then_schema = then_schema
        self.else_schema = else_schema

    def to_json_schema(self) -> dict:
        """Convert to JSON Schema if/then/else structure.

        Returns:
            JSON Schema conditional structure
        """
        schema = {'if': self.condition}

        if self.then_schema:
            schema['then'] = self.then_schema

        if self.else_schema:
            schema['else'] = self.else_schema

        return schema
