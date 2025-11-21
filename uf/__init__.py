"""uf - UI Fast: Minimal-boilerplate web UIs for Python functions.

uf bridges functions to HTTP services (via qh) to Web UI forms (via ju.rjsf),
following the "convention over configuration" philosophy.

Basic usage:
    >>> from uf import mk_rjsf_app
    >>>
    >>> def add(x: int, y: int) -> int:
    ...     '''Add two numbers'''
    ...     return x + y
    >>>
    >>> app = mk_rjsf_app([add])
    >>> # app.run()  # Start the web server

The main entry points are:
- `mk_rjsf_app`: Create a web app from functions (functional interface)
- `UfApp`: Object-oriented wrapper with additional conveniences
- `FunctionSpecStore`: Manage function specifications (advanced usage)

Advanced features:
- `ui_config`: Decorator for UI metadata
- `RjsfFieldConfig`: Field configuration class
- `FunctionGroup`: Group functions for organization
- `InputTransformRegistry`: Custom type transformations
- Field dependencies and interactions
- Testing utilities
"""

# Core functionality
from uf.base import mk_rjsf_app, UfApp
from uf.specs import FunctionSpecStore

# RJSF configuration
from uf.rjsf_config import (
    RjsfFieldConfig,
    RjsfConfigBuilder,
    get_field_config,
    apply_field_configs,
    ConditionalFieldConfig,
)

# Input transformation
from uf.trans import (
    InputTransformRegistry,
    register_type,
    get_global_registry,
)

# Organization
from uf.organization import (
    FunctionGroup,
    FunctionOrganizer,
    mk_grouped_app,
    auto_group_by_prefix,
    auto_group_by_module,
    auto_group_by_tag,
)

# Decorators
from uf.decorators import (
    ui_config,
    group,
    hidden,
    field_config,
    with_example,
    deprecated,
    requires_auth,
    rate_limit,
    get_ui_config,
    get_group,
    get_field_configs,
    is_hidden,
    get_example,
)

# Field interactions
from uf.field_interactions import (
    FieldDependency,
    DependencyAction,
    DependencyBuilder,
    with_dependencies,
    get_field_dependencies,
)

# Testing utilities
from uf.testing import (
    UfTestClient,
    UfAppTester,
    test_ui_function,
    FormDataBuilder,
    assert_valid_rjsf_spec,
    assert_has_field,
    assert_field_type,
    assert_field_required,
)

# Result rendering
from uf.renderers import (
    ResultRenderer,
    ResultRendererRegistry,
    get_global_renderer_registry,
    register_renderer,
    render_result,
    result_renderer,
    get_result_renderer,
)

# Async support
from uf.async_support import (
    is_async_function,
    async_to_sync,
    make_sync_compatible,
    AsyncFunctionWrapper,
    timeout_async,
    retry_async,
)

# Pydantic integration
from uf.pydantic_support import (
    is_pydantic_model,
    pydantic_model_to_json_schema,
    function_uses_pydantic,
    wrap_pydantic_function,
    pydantic_to_dict,
    dict_to_pydantic,
)

# History and presets
from uf.history import (
    FunctionCall,
    CallHistory,
    Preset,
    PresetManager,
    HistoryManager,
    get_global_history_manager,
    enable_history,
)

# Authentication
from uf.auth import (
    User,
    AuthBackend,
    DictAuthBackend,
    SessionManager,
    ApiKey,
    ApiKeyManager,
    require_auth,
    get_global_auth_backend,
)

# Caching
from uf.caching import (
    CacheBackend,
    MemoryCache,
    DiskCache,
    cached,
    get_global_cache_backend,
)

# Background tasks
from uf.background import (
    Task,
    TaskStatus,
    TaskQueue,
    background,
    get_global_task_queue,
)

# OpenAPI
from uf.openapi import (
    generate_openapi_spec,
    add_openapi_routes,
    OpenAPIConfig,
)

# Webhooks
from uf.webhooks import (
    WebhookEvent,
    WebhookClient,
    WebhookManager,
    webhook,
    get_global_webhook_manager,
)

# Themes
from uf.themes import (
    Theme,
    ThemeConfig,
    get_theme,
    LIGHT_THEME,
    DARK_THEME,
)

__version__ = "0.0.1"

__all__ = [
    # Core
    "mk_rjsf_app",
    "UfApp",
    "FunctionSpecStore",
    # RJSF Config
    "RjsfFieldConfig",
    "RjsfConfigBuilder",
    "get_field_config",
    "apply_field_configs",
    "ConditionalFieldConfig",
    # Transformation
    "InputTransformRegistry",
    "register_type",
    "get_global_registry",
    # Organization
    "FunctionGroup",
    "FunctionOrganizer",
    "mk_grouped_app",
    "auto_group_by_prefix",
    "auto_group_by_module",
    "auto_group_by_tag",
    # Decorators
    "ui_config",
    "group",
    "hidden",
    "field_config",
    "with_example",
    "deprecated",
    "requires_auth",
    "rate_limit",
    "get_ui_config",
    "get_group",
    "get_field_configs",
    "is_hidden",
    "get_example",
    # Field Interactions
    "FieldDependency",
    "DependencyAction",
    "DependencyBuilder",
    "with_dependencies",
    "get_field_dependencies",
    # Testing
    "UfTestClient",
    "UfAppTester",
    "test_ui_function",
    "FormDataBuilder",
    "assert_valid_rjsf_spec",
    "assert_has_field",
    "assert_field_type",
    "assert_field_required",
    # Renderers
    "ResultRenderer",
    "ResultRendererRegistry",
    "get_global_renderer_registry",
    "register_renderer",
    "render_result",
    "result_renderer",
    "get_result_renderer",
    # Async
    "is_async_function",
    "async_to_sync",
    "make_sync_compatible",
    "AsyncFunctionWrapper",
    "timeout_async",
    "retry_async",
    # Pydantic
    "is_pydantic_model",
    "pydantic_model_to_json_schema",
    "function_uses_pydantic",
    "wrap_pydantic_function",
    "pydantic_to_dict",
    "dict_to_pydantic",
    # History
    "FunctionCall",
    "CallHistory",
    "Preset",
    "PresetManager",
    "HistoryManager",
    "get_global_history_manager",
    "enable_history",
    # Authentication
    "User",
    "AuthBackend",
    "DictAuthBackend",
    "SessionManager",
    "ApiKey",
    "ApiKeyManager",
    "require_auth",
    "get_global_auth_backend",
    # Caching
    "CacheBackend",
    "MemoryCache",
    "DiskCache",
    "cached",
    "get_global_cache_backend",
    # Background
    "Task",
    "TaskStatus",
    "TaskQueue",
    "background",
    "get_global_task_queue",
    # OpenAPI
    "generate_openapi_spec",
    "add_openapi_routes",
    "OpenAPIConfig",
    # Webhooks
    "WebhookEvent",
    "WebhookClient",
    "WebhookManager",
    "webhook",
    "get_global_webhook_manager",
    # Themes
    "Theme",
    "ThemeConfig",
    "get_theme",
    "LIGHT_THEME",
    "DARK_THEME",
]
