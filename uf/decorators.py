"""UI metadata decorators for uf.

Provides decorators for annotating functions with UI configuration,
grouping, and field specifications.
"""

from typing import Callable, Optional, Any
from functools import wraps
from uf.rjsf_config import RjsfFieldConfig


# Attribute names for storing metadata
_UI_CONFIG_ATTR = '__uf_ui_config__'
_GROUP_ATTR = '__uf_group__'
_FIELD_CONFIGS_ATTR = '__uf_field_configs__'
_HIDDEN_ATTR = '__uf_hidden__'


def ui_config(
    *,
    title: Optional[str] = None,
    description: Optional[str] = None,
    group: Optional[str] = None,
    fields: Optional[dict[str, RjsfFieldConfig]] = None,
    hidden: bool = False,
    icon: Optional[str] = None,
    order: Optional[int] = None,
):
    """Decorator to add UI configuration to functions.

    This decorator attaches metadata to functions that uf can use to
    customize the generated UI.

    Args:
        title: Custom title for the function in UI
        description: Custom description (overrides docstring)
        group: Group name for organization
        fields: Dictionary mapping parameter names to RjsfFieldConfig
        hidden: Whether to hide this function from UI
        icon: Icon identifier for the function
        order: Display order within group

    Returns:
        Decorator function

    Example:
        >>> @ui_config(
        ...     title="User Registration",
        ...     group="Admin",
        ...     fields={'email': RjsfFieldConfig(format='email')}
        ... )
        ... def register_user(email: str, name: str):
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        config = {
            'title': title,
            'description': description,
            'group': group,
            'fields': fields or {},
            'hidden': hidden,
            'icon': icon,
            'order': order if order is not None else 0,
        }

        setattr(func, _UI_CONFIG_ATTR, config)

        # Also set group attribute for auto-grouping
        if group:
            setattr(func, _GROUP_ATTR, group)

        # Set hidden attribute
        if hidden:
            setattr(func, _HIDDEN_ATTR, True)

        # Set field configs
        if fields:
            setattr(func, _FIELD_CONFIGS_ATTR, fields)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Copy metadata to wrapper
        setattr(wrapper, _UI_CONFIG_ATTR, config)
        if group:
            setattr(wrapper, _GROUP_ATTR, group)
        if hidden:
            setattr(wrapper, _HIDDEN_ATTR, True)
        if fields:
            setattr(wrapper, _FIELD_CONFIGS_ATTR, fields)

        return wrapper

    return decorator


def group(group_name: str):
    """Decorator to assign a function to a group.

    Simpler alternative to ui_config when you only need to set the group.

    Args:
        group_name: Name of the group

    Returns:
        Decorator function

    Example:
        >>> @group("Admin")
        ... def delete_user(user_id: int):
        ...     pass
    """
    return ui_config(group=group_name)


def hidden(func: Callable) -> Callable:
    """Decorator to hide a function from the UI.

    The function will still be callable via the API but won't appear
    in the UI navigation.

    Args:
        func: Function to hide

    Returns:
        Decorated function

    Example:
        >>> @hidden
        ... def internal_function():
        ...     pass
    """
    return ui_config(hidden=True)(func)


def field_config(**field_configs: RjsfFieldConfig):
    """Decorator to configure specific fields of a function.

    Args:
        **field_configs: Keyword arguments mapping parameter names to configs

    Returns:
        Decorator function

    Example:
        >>> from uf.rjsf_config import get_field_config
        >>>
        >>> @field_config(
        ...     email=get_field_config('email'),
        ...     bio=get_field_config('multiline_text')
        ... )
        ... def create_profile(email: str, bio: str):
        ...     pass
    """
    return ui_config(fields=field_configs)


def get_ui_config(func: Callable) -> Optional[dict]:
    """Get UI configuration from a function.

    Args:
        func: Function to get config from

    Returns:
        Configuration dictionary or None if not configured

    Example:
        >>> config = get_ui_config(my_function)
        >>> if config:
        ...     print(config['title'])
    """
    return getattr(func, _UI_CONFIG_ATTR, None)


def get_group(func: Callable) -> Optional[str]:
    """Get group name from a function.

    Args:
        func: Function to get group from

    Returns:
        Group name or None

    Example:
        >>> group = get_group(my_function)
        >>> if group:
        ...     print(f"Function is in group: {group}")
    """
    config = get_ui_config(func)
    if config:
        return config.get('group')
    return getattr(func, _GROUP_ATTR, None)


def get_field_configs(func: Callable) -> dict[str, RjsfFieldConfig]:
    """Get field configurations from a function.

    Args:
        func: Function to get field configs from

    Returns:
        Dictionary mapping parameter names to RjsfFieldConfig

    Example:
        >>> field_configs = get_field_configs(my_function)
        >>> if 'email' in field_configs:
        ...     print(field_configs['email'].format)
    """
    config = get_ui_config(func)
    if config:
        return config.get('fields', {})
    return getattr(func, _FIELD_CONFIGS_ATTR, {})


def is_hidden(func: Callable) -> bool:
    """Check if a function is hidden from UI.

    Args:
        func: Function to check

    Returns:
        True if function is hidden, False otherwise

    Example:
        >>> if not is_hidden(my_function):
        ...     # Show in UI
        ...     pass
    """
    config = get_ui_config(func)
    if config:
        return config.get('hidden', False)
    return getattr(func, _HIDDEN_ATTR, False)


def with_example(*example_args, **example_kwargs):
    """Decorator to attach example arguments to a function.

    This can be used to provide example/test data that appears in the UI.

    Args:
        *example_args: Example positional arguments
        **example_kwargs: Example keyword arguments

    Returns:
        Decorator function

    Example:
        >>> @with_example(x=10, y=20)
        ... def add(x: int, y: int) -> int:
        ...     return x + y
    """

    def decorator(func: Callable) -> Callable:
        setattr(func, '__uf_example_args__', example_args)
        setattr(func, '__uf_example_kwargs__', example_kwargs)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(wrapper, '__uf_example_args__', example_args)
        setattr(wrapper, '__uf_example_kwargs__', example_kwargs)

        return wrapper

    return decorator


def get_example(func: Callable) -> Optional[tuple[tuple, dict]]:
    """Get example arguments from a function.

    Args:
        func: Function to get example from

    Returns:
        Tuple of (args, kwargs) or None if no example

    Example:
        >>> example = get_example(my_function)
        >>> if example:
        ...     args, kwargs = example
        ...     result = my_function(*args, **kwargs)
    """
    args = getattr(func, '__uf_example_args__', None)
    kwargs = getattr(func, '__uf_example_kwargs__', None)

    if args is not None or kwargs is not None:
        return (args or (), kwargs or {})

    return None


def deprecated(message: Optional[str] = None):
    """Decorator to mark a function as deprecated.

    Args:
        message: Optional deprecation message

    Returns:
        Decorator function

    Example:
        >>> @deprecated("Use new_function instead")
        ... def old_function():
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        setattr(func, '__uf_deprecated__', True)
        setattr(func, '__uf_deprecated_message__', message)

        @wraps(func)
        def wrapper(*args, **kwargs):
            import warnings

            msg = message or f"{func.__name__} is deprecated"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        setattr(wrapper, '__uf_deprecated__', True)
        setattr(wrapper, '__uf_deprecated_message__', message)

        return wrapper

    return decorator


def requires_auth(
    *,
    roles: Optional[list[str]] = None,
    permissions: Optional[list[str]] = None,
):
    """Decorator to mark a function as requiring authentication.

    This is metadata-only; actual authentication must be implemented
    separately in the application layer.

    Args:
        roles: List of required roles
        permissions: List of required permissions

    Returns:
        Decorator function

    Example:
        >>> @requires_auth(roles=['admin'], permissions=['user:delete'])
        ... def delete_user(user_id: int):
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        setattr(func, '__uf_requires_auth__', True)
        setattr(func, '__uf_required_roles__', roles or [])
        setattr(func, '__uf_required_permissions__', permissions or [])

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(wrapper, '__uf_requires_auth__', True)
        setattr(wrapper, '__uf_required_roles__', roles or [])
        setattr(wrapper, '__uf_required_permissions__', permissions or [])

        return wrapper

    return decorator


def rate_limit(calls: int, period: int):
    """Decorator to mark a function with rate limiting metadata.

    This is metadata-only; actual rate limiting must be implemented
    separately.

    Args:
        calls: Number of calls allowed
        period: Time period in seconds

    Returns:
        Decorator function

    Example:
        >>> @rate_limit(calls=10, period=60)  # 10 calls per minute
        ... def send_email(to: str, subject: str):
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        setattr(func, '__uf_rate_limit__', {'calls': calls, 'period': period})

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(wrapper, '__uf_rate_limit__', {'calls': calls, 'period': period})

        return wrapper

    return decorator
