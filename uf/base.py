"""Core functionality for uf - UI Fast.

This module provides the main `mk_rjsf_app` function that bridges:
- Functions → HTTP services (via qh)
- HTTP services → Web UI forms (via ju.rjsf)

Following the "convention over configuration" philosophy, it provides sensible
defaults while allowing customization where needed.
"""

from typing import Callable, Iterable, Optional, Any, Mapping as MappingType
from uf.specs import FunctionSpecStore
from uf.routes import add_ui_routes


def mk_rjsf_app(
    funcs: Iterable[Callable],
    *,
    # qh-related kwargs
    config: Optional[Any] = None,
    input_trans: Optional[Callable] = None,
    output_trans: Optional[Callable] = None,
    # rjsf-related kwargs
    rjsf_config: Optional[dict] = None,
    ui_schema_factory: Optional[Callable] = None,
    param_to_prop_type: Optional[Callable] = None,
    # uf-specific kwargs
    page_title: str = "Function Interface",
    function_display_names: Optional[MappingType] = None,
    custom_css: Optional[str] = None,
    rjsf_theme: str = "default",
    add_ui: bool = True,
    **qh_kwargs,
):
    """Create an RJSF-backed web app from a list of functions.

    This is the main entry point for uf. It combines qh's HTTP service
    generation with ju's RJSF form generation to create a complete
    web interface for your functions.

    Args:
        funcs: Iterable of callable functions to expose via web UI
        config: Optional qh.AppConfig for HTTP service configuration
        input_trans: Optional input transformation function for qh
        output_trans: Optional output transformation function for qh
        rjsf_config: Optional configuration dict for RJSF generation
        ui_schema_factory: Optional callable to customize UI schema
        param_to_prop_type: Optional callable to map parameters to types
        page_title: Title for the web interface
        function_display_names: Optional mapping to override function names
        custom_css: Optional custom CSS for the web interface
        rjsf_theme: RJSF theme to use ('default', 'material-ui', etc.)
        add_ui: Whether to add UI routes (set False for API-only)
        **qh_kwargs: Additional keyword arguments passed to qh.mk_app

    Returns:
        A configured web application (bottle or FastAPI app) with:
        - HTTP endpoints for each function
        - RJSF-based web interface (if add_ui=True)
        - API routes for function specs

    Example:
        >>> def add(x: int, y: int) -> int:
        ...     '''Add two numbers'''
        ...     return x + y
        ...
        >>> def greet(name: str) -> str:
        ...     '''Greet a person'''
        ...     return f"Hello, {name}!"
        ...
        >>> app = mk_rjsf_app([add, greet])
        >>> # app.run()  # Start the web server

    Example with customization:
        >>> from uf import mk_rjsf_app, RjsfFieldConfig
        >>>
        >>> def send_email(to: str, subject: str, body: str):
        ...     '''Send an email'''
        ...     pass
        ...
        >>> app = mk_rjsf_app(
        ...     [send_email],
        ...     page_title="Email Sender",
        ...     custom_css="body { background: #f0f0f0; }",
        ... )
    """
    # Convert to list to allow multiple iterations
    funcs = list(funcs)

    # Create function specification store
    function_specs = FunctionSpecStore(
        funcs,
        rjsf_config=rjsf_config or {},
        ui_schema_factory=ui_schema_factory,
        param_to_prop_type=param_to_prop_type,
    )

    # Create HTTP service using qh
    try:
        from qh import mk_app
    except ImportError:
        raise ImportError(
            "qh is required for mk_rjsf_app. Install it with: pip install qh"
        )

    # Build the qh app with function endpoints
    app = mk_app(
        funcs,
        config=config,
        input_trans=input_trans,
        output_trans=output_trans,
        **qh_kwargs,
    )

    # Store function_specs on the app for later access
    app.function_specs = function_specs

    # Add UI routes if requested
    if add_ui:
        add_ui_routes(
            app,
            function_specs,
            page_title=page_title,
            custom_css=custom_css,
            rjsf_theme=rjsf_theme,
        )

    return app


class UfApp:
    """Wrapper class for uf applications.

    Provides a higher-level interface with additional conveniences
    beyond the raw qh app.

    Attributes:
        app: The underlying qh/bottle/fastapi app
        function_specs: FunctionSpecStore for function metadata
        funcs: Dictionary mapping function names to callables
    """

    def __init__(
        self,
        funcs: Iterable[Callable],
        **mk_rjsf_app_kwargs,
    ):
        """Initialize UfApp.

        Args:
            funcs: Iterable of callable functions
            **mk_rjsf_app_kwargs: Arguments passed to mk_rjsf_app
        """
        self.funcs = {f.__name__: f for f in funcs}
        self.app = mk_rjsf_app(list(self.funcs.values()), **mk_rjsf_app_kwargs)
        self.function_specs = self.app.function_specs

    def run(self, host: str = 'localhost', port: int = 8080, **kwargs):
        """Run the web application.

        Args:
            host: Host to bind to
            port: Port to listen on
            **kwargs: Additional arguments passed to app.run()
        """
        if hasattr(self.app, 'run'):
            # Bottle app
            self.app.run(host=host, port=port, **kwargs)
        else:
            # FastAPI or other - provide guidance
            raise NotImplementedError(
                "For FastAPI apps, use: uvicorn.run(app.app, host='...', port=...)"
            )

    def call(self, func_name: str, **kwargs) -> Any:
        """Call a function directly by name.

        Args:
            func_name: Name of the function to call
            **kwargs: Arguments to pass to the function

        Returns:
            Result of the function call

        Raises:
            KeyError: If function name not found
        """
        if func_name not in self.funcs:
            raise KeyError(f"Function '{func_name}' not found")
        return self.funcs[func_name](**kwargs)

    def get_spec(self, func_name: str) -> dict:
        """Get RJSF specification for a function.

        Args:
            func_name: Name of the function

        Returns:
            Dictionary with schema and uiSchema

        Raises:
            KeyError: If function name not found
        """
        return self.function_specs[func_name]

    def list_functions(self) -> list[str]:
        """Get list of all function names.

        Returns:
            List of function name strings
        """
        return list(self.funcs.keys())

    def __repr__(self):
        """String representation of UfApp."""
        func_names = ', '.join(self.list_functions())
        return f"UfApp({func_names})"
