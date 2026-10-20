"""Function grouping and organization for uf.

Provides tools for organizing functions into categories, groups, and
hierarchies for better navigation in the UI.
"""

import functools
from typing import Any, Callable, Iterable, Optional, Union
from dataclasses import dataclass, field
from collections.abc import Mapping

from uf.decorators import attr_through_partials
from uf.specs import doc_of_func, name_funcs, name_of_func, unwrap_partials


@dataclass
class FunctionGroup:
    """Group of functions with metadata.

    Attributes:
        name: Name of the group
        funcs: Functions in this group: callables (partials and callable instances
            are fine; they are named as ``uf.specs.name_of_func`` does), or a
            ``{name: callable}`` mapping to name them explicitly
        description: Description of the group
        icon: Optional icon identifier for the group
        order: Display order (lower numbers first)
        collapsed: Whether the group starts collapsed in UI

    Example:
        >>> def add(x: int, y: int) -> int:
        ...     return x + y
        >>> def subtract(x: int, y: int) -> int:
        ...     return x - y
        >>> math_funcs = FunctionGroup(
        ...     'Math',
        ...     [add, subtract],
        ...     description='Mathematical operations',
        ...     icon='calculator'
        ... )
        >>> list(math_funcs.named_funcs)
        ['add', 'subtract']
    """

    name: str
    funcs: Union[list[Callable], Mapping[str, Callable]] = field(default_factory=list)
    description: str = ""
    icon: Optional[str] = None
    order: int = 0
    collapsed: bool = False

    @property
    def named_funcs(self) -> dict[str, Callable]:
        """``{name: callable}`` of this group, named the way the app serves them.

        >>> import functools
        >>> g = FunctionGroup('G', [len, functools.partial(int, base=2)])
        >>> list(g.named_funcs)
        ['len', 'int']
        """
        return name_funcs(self.funcs)

    def __post_init__(self):
        # (materialize now: a generator would be used up by the first read)
        self.funcs = (
            dict(self.funcs) if isinstance(self.funcs, Mapping) else list(self.funcs)
        )

    def add_function(
        self, func: Callable, *, name: Optional[str] = None
    ) -> "FunctionGroup":
        """Add a function to this group.

        Args:
            func: Function to add
            name: Explicit name to serve it under (default: its own name).
                Giving one turns ``funcs`` into a ``{name: callable}`` mapping.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If a different callable already has that name in the group
        """
        new_name = name if name is not None else name_of_func(func)
        existing = self.named_funcs.get(new_name)
        if existing is not None and existing != func:
            raise ValueError(
                f"group {self.name!r} already has a different callable named "
                f"{new_name!r}; pass name= to name this one explicitly"
            )
        if name is not None:
            self.funcs = {**self.named_funcs, name: func}
        elif isinstance(self.funcs, Mapping):
            self.funcs = {**self.funcs, new_name: func}
        else:
            self.funcs.append(func)
        return self

    def get_function_names(self) -> list[str]:
        """Get names of all functions in this group.

        Returns:
            List of function name strings
        """
        return list(self.named_funcs)


class FunctionOrganizer:
    """Organize functions into groups and hierarchies.

    This class provides a fluent interface for building function
    organization structures that can be used to generate grouped
    navigation in the UI.

    Example:
        >>> def user_create(name: str): ...
        >>> def user_delete(name: str): ...
        >>> def generate_report(): ...
        >>> organizer = FunctionOrganizer()
        >>> g = organizer.group('Admin', [user_create, user_delete], icon='shield')
        >>> g = organizer.group('Reports', [generate_report], icon='file')
        >>> [group.name for group in organizer.get_groups()]
        ['Admin', 'Reports']
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
            >>> def save_record(record: dict): ...
            >>> def load_record(key: str): ...
            >>> organizer = FunctionOrganizer()
            >>> group = organizer.group(
            ...     'Database',
            ...     [save_record, load_record],
            ...     description='Database operations',
            ...     icon='database'
            ... )
            >>> group.icon, list(group.named_funcs)
            ('database', ['save_record', 'load_record'])
        """
        func_group = FunctionGroup(
            name=name,
            funcs=dict(funcs) if isinstance(funcs, Mapping) else list(funcs or []),
            description=description,
            icon=icon,
            order=order,
            collapsed=collapsed,
        )
        self._groups.append(func_group)
        return func_group

    def add_to_group(self, group_name: str, func: Callable) -> "FunctionOrganizer":
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

    def add_ungrouped(self, func: Callable) -> "FunctionOrganizer":
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
            all_funcs.extend(group.named_funcs.values())
        all_funcs.extend(self._ungrouped_funcs)
        return all_funcs

    def to_dict(self) -> dict:
        """Convert organization to dictionary format.

        Returns:
            Dictionary representation suitable for JSON serialization

        Example:
            >>> def ping(): ...
            >>> organizer = FunctionOrganizer()
            >>> g = organizer.group('Ops', [ping])
            >>> organizer.to_dict()['groups'][0]['functions']
            [{'name': 'ping', 'description': ''}]
        """
        return {
            "groups": [
                {
                    "name": group.name,
                    "description": group.description,
                    "icon": group.icon,
                    "order": group.order,
                    "collapsed": group.collapsed,
                    "functions": [
                        {
                            "name": name,
                            "description": doc_of_func(func) or "",
                        }
                        for name, func in group.named_funcs.items()
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
        >>> def user_create(name: str): ...
        >>> def generate_report(): ...
        >>> admin_group = FunctionGroup('Admin', [user_create])
        >>> reports_group = FunctionGroup('Reports', [generate_report])
        >>> app = mk_grouped_app([admin_group, reports_group])
        >>> type(app).__name__
        'FastAPI'
    """
    from uf.base import mk_rjsf_app

    groups = list(groups)

    # Collect all functions from all groups, under the names the groups show
    all_funcs = {}
    for group in groups:
        for name, func in group.named_funcs.items():
            if name in all_funcs and all_funcs[name] != func:
                raise ValueError(
                    f"two different callables are both named {name!r} (group "
                    f"{group.name!r}); pass a {{name: callable}} mapping as the "
                    "group's funcs to name them explicitly"
                )
            all_funcs[name] = func

    # Create the app
    app = mk_rjsf_app(all_funcs, **mk_rjsf_app_kwargs)

    # Store organization metadata on the app
    app.function_groups = groups

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
    is_bottle = hasattr(app, "route")

    if is_bottle:

        @app.route("/api/groups")
        def get_groups():
            """Get function group organization."""
            import json
            from bottle import response

            response.content_type = "application/json"
            return json.dumps(organizer.to_dict())
    else:
        # FastAPI
        from fastapi.responses import JSONResponse

        @app.get("/api/groups")
        async def get_groups():
            """Get function group organization."""
            return JSONResponse(content=organizer.to_dict())


_attr_through_partials = attr_through_partials  # backward-compatible alias


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
        >>> def user_create(): ...
        >>> def user_delete(): ...
        >>> def report_generate(): ...
        >>> def admin_reset(): ...
        >>> funcs = [user_create, user_delete, report_generate, admin_reset]
        >>> organizer = auto_group_by_prefix(funcs)
        >>> [group.name for group in organizer.get_groups()]
        ['Admin', 'Report', 'User']
    """
    from collections import defaultdict

    # Group functions by prefix
    groups_dict = defaultdict(list)

    for func in funcs:
        name = name_of_func(func)
        if separator in name:
            prefix = name.split(separator)[0]
            groups_dict[prefix].append(func)
        else:
            groups_dict["other"].append(func)

    # Create organizer
    organizer = FunctionOrganizer()

    for group_name, group_funcs in sorted(groups_dict.items()):
        # Capitalize group name
        display_name = group_name.replace("_", " ").title()

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
        FunctionOrganizer with module-based groups (a ``functools.partial`` goes with the
        module of the function it wraps)

    Example:
        >>> def create(): ...
        >>> create.__module__ = 'myapp.user_ops'
        >>> def generate(): ...
        >>> generate.__module__ = 'myapp.report_ops'
        >>> organizer = auto_group_by_module([create, generate])
        >>> [group.name for group in organizer.get_groups()]
        ['Report Ops', 'User Ops']
    """
    from collections import defaultdict

    groups_dict = defaultdict(list)

    for func in funcs:
        module = unwrap_partials(func).__module__
        # Get last part of module name
        if "." in module:
            module_name = module.split(".")[-1]
        else:
            module_name = module

        groups_dict[module_name].append(func)

    # Create organizer
    organizer = FunctionOrganizer()

    for module_name, group_funcs in sorted(groups_dict.items()):
        display_name = module_name.replace("_", " ").title()

        organizer.group(
            display_name,
            group_funcs,
            description=f"Functions from {module_name}",
        )

    return organizer


def auto_group_by_tag(
    funcs: Iterable[Callable], tag_attr: str = "__uf_group__"
) -> FunctionOrganizer:
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
        >>> def other_func(): ...
        >>> organizer = auto_group_by_tag([create_user, other_func])
        >>> [group.name for group in organizer.get_groups()]
        ['Admin', 'Other']
    """
    from collections import defaultdict

    groups_dict = defaultdict(list)

    for func in funcs:
        tag = _attr_through_partials(func, tag_attr, "Other")
        groups_dict[tag].append(func)

    # Create organizer
    organizer = FunctionOrganizer()

    for tag, group_funcs in sorted(groups_dict.items()):
        organizer.group(tag, group_funcs)

    return organizer
