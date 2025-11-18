"""Field dependencies and interactions for uf.

Provides tools for defining relationships between form fields, such as
conditional display, dynamic validation, and field dependencies.
"""

from typing import Callable, Optional, Any, Literal
from dataclasses import dataclass
from enum import Enum


class DependencyAction(Enum):
    """Actions that can be triggered by field dependencies."""

    SHOW = "show"  # Show the target field
    HIDE = "hide"  # Hide the target field
    ENABLE = "enable"  # Enable the target field
    DISABLE = "disable"  # Disable the target field
    REQUIRE = "require"  # Make the target field required
    OPTIONAL = "optional"  # Make the target field optional


@dataclass
class FieldDependency:
    """Define a dependency between form fields.

    Attributes:
        source_field: Name of the field that triggers the dependency
        target_field: Name of the field affected by the dependency
        condition: Function that takes source value and returns bool
        action: Action to perform when condition is true
        else_action: Optional action when condition is false

    Example:
        >>> # Show 'other_reason' only when reason is 'other'
        >>> dep = FieldDependency(
        ...     source_field='reason',
        ...     target_field='other_reason',
        ...     condition=lambda v: v == 'other',
        ...     action=DependencyAction.SHOW,
        ...     else_action=DependencyAction.HIDE
        ... )
    """

    source_field: str
    target_field: str
    condition: Callable[[Any], bool]
    action: DependencyAction
    else_action: Optional[DependencyAction] = None

    def check(self, value: Any) -> DependencyAction:
        """Check the condition and return the appropriate action.

        Args:
            value: Value of the source field

        Returns:
            The action to perform
        """
        if self.condition(value):
            return self.action
        elif self.else_action:
            return self.else_action
        else:
            # Return opposite action if no else_action specified
            opposites = {
                DependencyAction.SHOW: DependencyAction.HIDE,
                DependencyAction.HIDE: DependencyAction.SHOW,
                DependencyAction.ENABLE: DependencyAction.DISABLE,
                DependencyAction.DISABLE: DependencyAction.ENABLE,
                DependencyAction.REQUIRE: DependencyAction.OPTIONAL,
                DependencyAction.OPTIONAL: DependencyAction.REQUIRE,
            }
            return opposites.get(self.action, self.action)

    def to_json_schema(self) -> dict:
        """Convert to JSON Schema dependencies format.

        Returns:
            JSON Schema dependencies structure

        Note:
            This uses JSON Schema's if/then/else structure for dependencies.
        """
        # Build condition schema
        if callable(self.condition):
            # For callable conditions, we need to handle specific cases
            # This is a simplified version - complex conditions may need custom handling
            condition_schema = {'properties': {self.source_field: {}}}
        else:
            condition_schema = self.condition

        # Build then/else schemas based on actions
        then_schema = self._action_to_schema(self.action)
        else_schema = (
            self._action_to_schema(self.else_action) if self.else_action else None
        )

        result = {'if': condition_schema}

        if then_schema:
            result['then'] = then_schema

        if else_schema:
            result['else'] = else_schema

        return result

    def _action_to_schema(self, action: DependencyAction) -> dict:
        """Convert action to JSON Schema modification.

        Args:
            action: The dependency action

        Returns:
            Schema modification dict
        """
        if action == DependencyAction.SHOW:
            # In JSON Schema, showing is the default, hiding uses uiSchema
            return {}
        elif action == DependencyAction.HIDE:
            return {}
        elif action == DependencyAction.REQUIRE:
            return {'required': [self.target_field]}
        elif action == DependencyAction.OPTIONAL:
            return {}
        else:
            return {}

    def to_ui_schema(self) -> dict:
        """Convert to RJSF UI Schema dependencies format.

        Returns:
            UI Schema dependencies structure
        """
        # UI Schema handles widget-level interactions
        ui_deps = {}

        if self.action == DependencyAction.HIDE:
            ui_deps[self.target_field] = {'ui:widget': 'hidden'}
        elif self.action == DependencyAction.DISABLE:
            ui_deps[self.target_field] = {'ui:disabled': True}

        return ui_deps


class DependencyBuilder:
    """Builder for creating field dependencies.

    Provides a fluent interface for defining dependencies between fields.

    Example:
        >>> builder = DependencyBuilder()
        >>> builder.when('reason').equals('other').show('other_reason')
        >>> builder.when('age').greater_than(18).enable('alcohol_consent')
        >>> dependencies = builder.build()
    """

    def __init__(self):
        """Initialize the dependency builder."""
        self._dependencies: list[FieldDependency] = []
        self._current_field: Optional[str] = None
        self._current_condition: Optional[Callable] = None

    def when(self, field_name: str) -> 'DependencyBuilder':
        """Start a dependency condition on a field.

        Args:
            field_name: Name of the source field

        Returns:
            Self for method chaining
        """
        self._current_field = field_name
        self._current_condition = None
        return self

    def equals(self, value: Any) -> 'DependencyBuilder':
        """Condition: field equals value.

        Args:
            value: Value to compare against

        Returns:
            Self for method chaining
        """
        self._current_condition = lambda v: v == value
        return self

    def not_equals(self, value: Any) -> 'DependencyBuilder':
        """Condition: field does not equal value.

        Args:
            value: Value to compare against

        Returns:
            Self for method chaining
        """
        self._current_condition = lambda v: v != value
        return self

    def greater_than(self, value: Any) -> 'DependencyBuilder':
        """Condition: field is greater than value.

        Args:
            value: Value to compare against

        Returns:
            Self for method chaining
        """
        self._current_condition = lambda v: v > value
        return self

    def less_than(self, value: Any) -> 'DependencyBuilder':
        """Condition: field is less than value.

        Args:
            value: Value to compare against

        Returns:
            Self for method chaining
        """
        self._current_condition = lambda v: v < value
        return self

    def is_truthy(self) -> 'DependencyBuilder':
        """Condition: field is truthy.

        Returns:
            Self for method chaining
        """
        self._current_condition = lambda v: bool(v)
        return self

    def is_falsy(self) -> 'DependencyBuilder':
        """Condition: field is falsy.

        Returns:
            Self for method chaining
        """
        self._current_condition = lambda v: not bool(v)
        return self

    def in_list(self, values: list) -> 'DependencyBuilder':
        """Condition: field value is in list.

        Args:
            values: List of acceptable values

        Returns:
            Self for method chaining
        """
        self._current_condition = lambda v: v in values
        return self

    def custom(self, condition: Callable[[Any], bool]) -> 'DependencyBuilder':
        """Condition: custom callable.

        Args:
            condition: Function that takes value and returns bool

        Returns:
            Self for method chaining
        """
        self._current_condition = condition
        return self

    def show(self, target_field: str) -> 'DependencyBuilder':
        """Action: show target field.

        Args:
            target_field: Name of field to show

        Returns:
            Self for method chaining
        """
        return self._add_dependency(target_field, DependencyAction.SHOW)

    def hide(self, target_field: str) -> 'DependencyBuilder':
        """Action: hide target field.

        Args:
            target_field: Name of field to hide

        Returns:
            Self for method chaining
        """
        return self._add_dependency(target_field, DependencyAction.HIDE)

    def enable(self, target_field: str) -> 'DependencyBuilder':
        """Action: enable target field.

        Args:
            target_field: Name of field to enable

        Returns:
            Self for method chaining
        """
        return self._add_dependency(target_field, DependencyAction.ENABLE)

    def disable(self, target_field: str) -> 'DependencyBuilder':
        """Action: disable target field.

        Args:
            target_field: Name of field to disable

        Returns:
            Self for method chaining
        """
        return self._add_dependency(target_field, DependencyAction.DISABLE)

    def require(self, target_field: str) -> 'DependencyBuilder':
        """Action: make target field required.

        Args:
            target_field: Name of field to require

        Returns:
            Self for method chaining
        """
        return self._add_dependency(target_field, DependencyAction.REQUIRE)

    def make_optional(self, target_field: str) -> 'DependencyBuilder':
        """Action: make target field optional.

        Args:
            target_field: Name of field to make optional

        Returns:
            Self for method chaining
        """
        return self._add_dependency(target_field, DependencyAction.OPTIONAL)

    def _add_dependency(
        self, target_field: str, action: DependencyAction
    ) -> 'DependencyBuilder':
        """Add a dependency to the list.

        Args:
            target_field: Target field name
            action: Action to perform

        Returns:
            Self for method chaining
        """
        if not self._current_field or not self._current_condition:
            raise ValueError("Must call when() and a condition method first")

        dep = FieldDependency(
            source_field=self._current_field,
            target_field=target_field,
            condition=self._current_condition,
            action=action,
        )

        self._dependencies.append(dep)
        return self

    def build(self) -> list[FieldDependency]:
        """Build and return the list of dependencies.

        Returns:
            List of FieldDependency objects
        """
        return self._dependencies.copy()


def add_field_dependencies(
    func: Callable,
    dependencies: list[FieldDependency],
) -> dict:
    """Augment RJSF spec with field dependencies.

    Args:
        func: Function to augment
        dependencies: List of field dependencies

    Returns:
        Dictionary with augmented schema and uiSchema

    Example:
        >>> deps = [
        ...     FieldDependency('reason', 'other_reason',
        ...                    lambda v: v == 'other',
        ...                    DependencyAction.SHOW)
        ... ]
        >>> spec = add_field_dependencies(my_func, deps)
    """
    # This would typically be called during spec generation
    # For now, return a structure that can be merged
    schema_additions = {'allOf': []}
    ui_schema_additions = {}

    for dep in dependencies:
        # Add JSON Schema dependencies
        schema_dep = dep.to_json_schema()
        if schema_dep:
            schema_additions['allOf'].append(schema_dep)

        # Add UI Schema dependencies
        ui_dep = dep.to_ui_schema()
        if ui_dep:
            ui_schema_additions.update(ui_dep)

    return {'schema_additions': schema_additions, 'ui_schema_additions': ui_schema_additions}


def with_dependencies(*dependencies: FieldDependency):
    """Decorator to attach field dependencies to a function.

    Args:
        *dependencies: FieldDependency objects

    Returns:
        Decorator function

    Example:
        >>> @with_dependencies(
        ...     FieldDependency('reason', 'other_reason',
        ...                    lambda v: v == 'other',
        ...                    DependencyAction.SHOW)
        ... )
        ... def submit_feedback(reason: str, other_reason: str = ''):
        ...     pass
    """
    from functools import wraps

    def decorator(func: Callable) -> Callable:
        setattr(func, '__uf_field_dependencies__', list(dependencies))

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(wrapper, '__uf_field_dependencies__', list(dependencies))

        return wrapper

    return decorator


def get_field_dependencies(func: Callable) -> list[FieldDependency]:
    """Get field dependencies from a function.

    Args:
        func: Function to get dependencies from

    Returns:
        List of FieldDependency objects
    """
    return getattr(func, '__uf_field_dependencies__', [])
