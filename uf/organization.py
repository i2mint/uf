"""Function grouping and organization for uf.

Provides tools for organizing functions into categories, groups, and
hierarchies for better navigation in the UI.
"""

from typing import Callable, Iterable, Optional
from dataclasses import dataclass, field
from collections.abc import Mapping


@dataclass
class FunctionGroup:
    """Group of functions with metadata.

    Attributes:
        name: Name of the group
        funcs: Functions in this group
        description: Description of the group
        icon: Optional icon identifier for the group
        order: Display order (lower numbers first)
        collapsed: Whether the group starts collapsed in UI

    Example:
        >>> math_funcs = FunctionGroup(
        ...     'Math',
        ...     [add, subtract, multiply, divide],
        ...     description='Mathematical operations',
        ...     icon='calculator'
        ... )
    """

    name: str
    funcs: list[Callable] = field(default_factory=list)
    description: str = ""
    icon: Optional[str] = None
    order: int = 0
    collapsed: bool = False

    def add_function(self, func: Callable) -> 'FunctionGroup':
        """Add a function to this group.

        Args:
            func: Function to add

        Returns:
            Self for method chaining
        """
        self.funcs.append(func)
        return self

    def get_function_names(self) -> list[str]:
        """Get names of all functions in this group.

        Returns:
            List of function name strings
        """
        return [f.__name__ for f in self.funcs]


class FunctionOrganizer:
    """Organize functions into groups and hierarchies.

    This class provides a fluent interface for building function
    organization structures that can be used to generate grouped
    navigation in the UI.

    Example:
        >>> organizer = FunctionOrganizer()
        >>> organizer.group('Admin', [user_create, user_delete], icon='shield')
        >>> organizer.group('Reports', [generate_report, export_csv], icon='file')
        >>> groups = organizer.get_groups()
    """

    def __init__(self):
        """Initialize the organizer."""
        self._groups: list[FunctionGroup] = []
        self._ungrouped_funcs: list[Callable] = []

    def group(
        self,
        name: str,
        funcs: Optional[Iterable[Callable]] = None,
        *,
        description: str = "",
        icon: Optional[str] = None,
        order: int = 0,
        collapsed: bool = False,
    ) -> FunctionGroup:
        """Create and add a function group.

        Args:
            name: Name of the group
            funcs: Optional functions to add to group
            description: Description of the group
            icon: Optional icon identifier
            order: Display order
            collapsed: Whether to start collapsed

        Returns:
            The created FunctionGroup

        Example:
            >>> organizer.group(
            ...     'Database',
            ...     [save_record, load_record],
            ...     description='Database operations',
            ...     icon='database'
            ... )
        """
        func_group = FunctionGroup(
            name=name,
            funcs=list(funcs or []),
            description=description,
            icon=icon,
            order=order,
            collapsed=collapsed,
        )
        self._groups.append(func_group)
        return func_group

    def add_to_group(self, group_name: str, func: Callable) -> 'FunctionOrganizer':
        """Add a function to an existing group.

        Args:
            group_name: Name of the group
            func: Function to add

        Returns:
            Self for method chaining

        Raises:
            ValueError: If group doesn't exist
        """
        for group in self._groups:
            if group.name == group_name:
                group.add_function(func)
                return self

        raise ValueError(f"Group '{group_name}' not found")

    def add_ungrouped(self, func: Callable) -> 'FunctionOrganizer':
        """Add a function without a group.

        Args:
            func: Function to add

        Returns:
            Self for method chaining
        """
        self._ungrouped_funcs.append(func)
        return self

    def get_groups(self) -> list[FunctionGroup]:
        """Get all groups, sorted by order.

        Returns:
            List of FunctionGroup objects sorted by order
        """
        groups = sorted(self._groups, key=lambda g: g.order)

        # Add ungrouped functions if any
        if self._ungrouped_funcs:
            ungrouped = FunctionGroup(
                name="Other",
                funcs=self._ungrouped_funcs,
                description="Uncategorized functions",
                order=999,
            )
            groups.append(ungrouped)

        return groups

    def get_all_functions(self) -> list[Callable]:
        """Get all functions across all groups.

        Returns:
            List of all functions
        """
        all_funcs = []
        for group in self._groups:
            all_funcs.extend(group.funcs)
        all_funcs.extend(self._ungrouped_funcs)
        return all_funcs

    def to_dict(self) -> dict:
        """Convert organization to dictionary format.

        Returns:
            Dictionary representation suitable for JSON serialization

        Example:
            >>> org_dict = organizer.to_dict()
            >>> # Can be used in templates or APIs
        """
        return {
            'groups': [
                {
                    'name': group.name,
                    'description': group.description,
                    'icon': group.icon,
                    'order': group.order,
                    'collapsed': group.collapsed,
                    'functions': [
                        {
                            'name': func.__name__,
                            'description': func.__doc__ or '',
                        }
                        for func in group.funcs
                    ],
                }
                for group in self.get_groups()
            ]
        }


def mk_grouped_app(
    groups: Iterable[FunctionGroup],
    **mk_rjsf_app_kwargs,
):
    """Create a uf app with grouped function navigation.

    Args:
        groups: Iterable of FunctionGroup objects
        **mk_rjsf_app_kwargs: Arguments passed to mk_rjsf_app

    Returns:
        Configured web application with grouped navigation

    Example:
        >>> admin_group = FunctionGroup('Admin', [user_create, user_delete])
        >>> reports_group = FunctionGroup('Reports', [generate_report])
        >>> app = mk_grouped_app([admin_group, reports_group])
    """
    from uf.base import mk_rjsf_app

    # Collect all functions from all groups
    all_funcs = []
    for group in groups:
        all_funcs.extend(group.funcs)

    # Create the app
    app = mk_rjsf_app(all_funcs, **mk_rjsf_app_kwargs)

    # Store organization metadata on the app
    app.function_groups = list(groups)

    # Create organizer for serialization
    organizer = FunctionOrganizer()
    for group in groups:
        organizer.group(
            name=group.name,
            funcs=group.funcs,
            description=group.description,
            icon=group.icon,
            order=group.order,
            collapsed=group.collapsed,
        )

    app.organization = organizer

    # Add route to get group information
    _add_group_routes(app, organizer)

    return app


def _add_group_routes(app, organizer: FunctionOrganizer):
    """Add routes for accessing group information.

    Args:
        app: The web application
        organizer: FunctionOrganizer instance
    """
    # Detect framework
    is_bottle = hasattr(app, 'route')

    if is_bottle:
        @app.route('/api/groups')
        def get_groups():
            """Get function group organization."""
            import json
            from bottle import response

            response.content_type = 'application/json'
            return json.dumps(organizer.to_dict())
    else:
        # FastAPI
        from fastapi.responses import JSONResponse

        @app.get('/api/groups')
        async def get_groups():
            """Get function group organization."""
            return JSONResponse(content=organizer.to_dict())


def auto_group_by_prefix(
    funcs: Iterable[Callable],
    separator: str = "_",
) -> FunctionOrganizer:
    """Automatically group functions by name prefix.

    Groups functions based on the part of their name before the separator.
    For example, with separator="_":
    - user_create, user_delete → "user" group
    - report_generate, report_export → "report" group

    Args:
        funcs: Functions to organize
        separator: Separator character (default: "_")

    Returns:
        FunctionOrganizer with auto-generated groups

    Example:
        >>> funcs = [user_create, user_delete, report_generate, admin_reset]
        >>> organizer = auto_group_by_prefix(funcs)
        >>> # Creates groups: 'user', 'report', 'admin'
    """
    from collections import defaultdict

    # Group functions by prefix
    groups_dict = defaultdict(list)

    for func in funcs:
        name = func.__name__
        if separator in name:
            prefix = name.split(separator)[0]
            groups_dict[prefix].append(func)
        else:
            groups_dict['other'].append(func)

    # Create organizer
    organizer = FunctionOrganizer()

    for group_name, group_funcs in sorted(groups_dict.items()):
        # Capitalize group name
        display_name = group_name.replace('_', ' ').title()

        organizer.group(
            display_name,
            group_funcs,
            description=f"{display_name} operations",
        )

    return organizer


def auto_group_by_module(funcs: Iterable[Callable]) -> FunctionOrganizer:
    """Automatically group functions by their module.

    Args:
        funcs: Functions to organize

    Returns:
        FunctionOrganizer with module-based groups

    Example:
        >>> from myapp import user_ops, report_ops
        >>> funcs = [user_ops.create, user_ops.delete, report_ops.generate]
        >>> organizer = auto_group_by_module(funcs)
    """
    from collections import defaultdict

    groups_dict = defaultdict(list)

    for func in funcs:
        module = func.__module__
        # Get last part of module name
        if '.' in module:
            module_name = module.split('.')[-1]
        else:
            module_name = module

        groups_dict[module_name].append(func)

    # Create organizer
    organizer = FunctionOrganizer()

    for module_name, group_funcs in sorted(groups_dict.items()):
        display_name = module_name.replace('_', ' ').title()

        organizer.group(
            display_name,
            group_funcs,
            description=f"Functions from {module_name}",
        )

    return organizer


def auto_group_by_tag(funcs: Iterable[Callable], tag_attr: str = '__uf_group__') -> FunctionOrganizer:
    """Automatically group functions by a tag attribute.

    Functions can be tagged with a group name using an attribute.

    Args:
        funcs: Functions to organize
        tag_attr: Name of the attribute to use for grouping

    Returns:
        FunctionOrganizer with tag-based groups

    Example:
        >>> def create_user(name: str):
        ...     pass
        >>> create_user.__uf_group__ = 'Admin'
        >>>
        >>> organizer = auto_group_by_tag([create_user, other_func])
    """
    from collections import defaultdict

    groups_dict = defaultdict(list)

    for func in funcs:
        tag = getattr(func, tag_attr, 'Other')
        groups_dict[tag].append(func)

    # Create organizer
    organizer = FunctionOrganizer()

    for tag, group_funcs in sorted(groups_dict.items()):
        organizer.group(tag, group_funcs)

    return organizer
