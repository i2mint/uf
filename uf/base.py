"""Core functionality for uf - UI Fast.

This module provides the main `mk_rjsf_app` function that bridges:
- Functions → HTTP services (via qh)
- HTTP services → Web UI forms (via ju.rjsf)

Following the "convention over configuration" philosophy, it provides sensible
defaults while allowing customization where needed.
"""

import functools
import inspect
import typing
from typing import Callable, Iterable, Optional, Any, Union, Mapping as MappingType
from uf.optional import import_optional
from uf.specs import FunctionSpecStore, doc_of_func, name_funcs
from uf.routes import add_ui_routes


def _serve_as(func: Callable, name: str) -> Callable:
    """``func`` as something qh can route: a function whose ``__name__`` is ``name``.

    qh names a route after ``func.__name__``, which a ``functools.partial``, a callable
    instance, or a function served under another name (a ``{name: callable}`` mapping)
    doesn't have. Such a callable is wrapped in a function carrying the name and the
    signature, annotations (string ones resolved in the module of ``func``) and
    docstring of ``func``; a function, method, builtin or class already named ``name``
    is returned as it is. (So qh options that refer to a function, like
    ``async_funcs``, must use the served name when one was given.)

    >>> import functools
    >>> def plus(x: int, y: int = 1) -> int:
    ...     return x + y
    >>> served = _serve_as(functools.partial(plus, y=10), 'plus_ten')
    >>> served.__name__, str(inspect.signature(served)), served(x=1)
    ('plus_ten', '(x: int, *, y: int = 10) -> int', 11)
    >>> _serve_as(plus, 'plus') is plus
    True
    """
    if getattr(func, "__name__", None) == name and (
        inspect.isroutine(func) or inspect.isclass(func)
    ):
        return func
    inner = func  # what the signature, annotations and awaiting are those of
    while isinstance(inner, functools.partial):
        inner = inner.func
    target = inner if inspect.isroutine(inner) else getattr(type(inner), "__call__", inner)
    try:  # string annotations name things of the user's module, not of this one
        hints = typing.get_type_hints(target)
    except Exception:
        hints = {}
    sig = inspect.signature(func)
    sig = sig.replace(
        parameters=[
            p.replace(annotation=hints.get(p.name, p.annotation))
            for p in sig.parameters.values()
        ],
        return_annotation=hints.get("return", sig.return_annotation),
    )
    if inspect.iscoroutinefunction(target):

        async def served(*args, **kwargs):
            return await func(*args, **kwargs)

    else:

        def served(*args, **kwargs):
            return func(*args, **kwargs)

    served.__name__ = served.__qualname__ = name
    served.__doc__ = doc_of_func(func)
    served.__module__ = getattr(inner, "__module__", None) or served.__module__
    served.__signature__ = sig
    served.__annotations__ = {
        **{
            p.name: p.annotation
            for p in sig.parameters.values()
            if p.annotation is not inspect.Parameter.empty
        },
        **(
            {"return": sig.return_annotation}
            if sig.return_annotation is not inspect.Signature.empty
            else {}
        ),
    }
    return served


def mk_rjsf_app(
    funcs: Union[Iterable[Callable], MappingType[str, Callable]],
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
        funcs: Iterable of callables to expose via web UI -- functions, partials,
            callable instances (named by ``uf.specs.name_of_func``) -- or a mapping
            ``{name: callable}`` to name them yourself. The name is the route and the
            key of the form spec
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
    # Name the callables once; the spec store and qh are given the same names
    named_funcs = name_funcs(funcs)

    # Create function specification store
    function_specs = FunctionSpecStore(
        named_funcs,
        rjsf_config=rjsf_config or {},
        ui_schema_factory=ui_schema_factory,
        param_to_prop_type=param_to_prop_type,
        validate=True,  # a function uf can't build a form for fails here, not on a request
    )

    # Create HTTP service using qh
    qh = import_optional("qh")  # a qh that fails to import is raised as it is
    if qh is None:
        raise ImportError(
            "qh is required for mk_rjsf_app. Install it with: pip install qh"
        )
    mk_app = qh.mk_app

    # Build the qh app with function endpoints
    app = mk_app(
        [_serve_as(func, name) for name, func in named_funcs.items()],
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
        funcs: Union[Iterable[Callable], MappingType[str, Callable]],
        **mk_rjsf_app_kwargs,
    ):
        """Initialize UfApp.

        Args:
            funcs: Iterable of callables, or a mapping ``{name: callable}`` (see
                ``mk_rjsf_app``)
            **mk_rjsf_app_kwargs: Arguments passed to mk_rjsf_app
        """
        self.funcs = name_funcs(funcs)
        self.app = mk_rjsf_app(self.funcs, **mk_rjsf_app_kwargs)
        self.function_specs = self.app.function_specs

    def run(self, host: str = "localhost", port: int = 8080, **kwargs):
        """Run the web application.

        Args:
            host: Host to bind to
            port: Port to listen on
            **kwargs: Additional arguments passed to app.run()
        """
        if hasattr(self.app, "run"):
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
        func_names = ", ".join(self.list_functions())
        return f"UfApp({func_names})"
