"""API routes for uf web interface.

Provides convenience routes for the web UI, including function listing,
spec retrieval, and the main HTML interface.
"""

from typing import Any, Callable
from collections.abc import Mapping


def add_ui_routes(
    app: Any,
    function_specs: Mapping,
    *,
    page_title: str = "Function Interface",
    custom_css: str = None,
    rjsf_theme: str = "default",
) -> None:
    """Add UI routes to a qh app.

    Adds the following routes:
    - GET / : Main UI page (HTML)
    - GET /api/functions : List available functions (JSON)
    - GET /api/functions/{name}/spec : Get RJSF spec for function (JSON)

    Args:
        app: The qh/bottle/fastapi app to add routes to
        function_specs: FunctionSpecStore with function specifications
        page_title: Title for the web interface
        custom_css: Optional custom CSS
        rjsf_theme: RJSF theme to use

    Note:
        This function detects whether the app is using Bottle or FastAPI
        and adds routes accordingly.
    """
    from uf.templates import generate_index_html, generate_error_page

    # Detect framework
    is_bottle = hasattr(app, 'route')
    is_fastapi = hasattr(app, 'get')

    if is_bottle:
        _add_bottle_routes(
            app,
            function_specs,
            page_title=page_title,
            custom_css=custom_css,
            rjsf_theme=rjsf_theme,
        )
    elif is_fastapi:
        _add_fastapi_routes(
            app,
            function_specs,
            page_title=page_title,
            custom_css=custom_css,
            rjsf_theme=rjsf_theme,
        )
    else:
        raise ValueError(f"Unsupported app type: {type(app)}")


def _add_bottle_routes(
    app,
    function_specs: Mapping,
    *,
    page_title: str,
    custom_css: str,
    rjsf_theme: str,
) -> None:
    """Add routes for Bottle framework."""
    from uf.templates import generate_index_html

    @app.route('/')
    def index():
        """Serve main UI page."""
        try:
            html = generate_index_html(
                function_specs,
                page_title=page_title,
                custom_css=custom_css,
                rjsf_theme=rjsf_theme,
            )
            return html
        except Exception as e:
            from uf.templates import generate_error_page
            return generate_error_page(str(e), 500)

    @app.route('/api/functions')
    def list_functions():
        """List all available functions."""
        import json
        from bottle import response

        response.content_type = 'application/json'

        try:
            func_list = [
                {
                    'name': name,
                    'description': spec.get('description', ''),
                }
                for name, spec in function_specs.items()
            ]
            return json.dumps(func_list)
        except Exception as e:
            response.status = 500
            return json.dumps({'error': str(e)})

    @app.route('/api/functions/<func_name>/spec')
    def get_function_spec(func_name):
        """Get RJSF specification for a function."""
        import json
        from bottle import response

        response.content_type = 'application/json'

        try:
            spec = function_specs[func_name]
            return json.dumps({
                'schema': spec['schema'],
                'uiSchema': spec['uiSchema'],
                'name': spec['name'],
                'description': spec['description'],
            })
        except KeyError:
            response.status = 404
            return json.dumps({'error': f"Function '{func_name}' not found"})
        except Exception as e:
            response.status = 500
            return json.dumps({'error': str(e)})


def _add_fastapi_routes(
    app,
    function_specs: Mapping,
    *,
    page_title: str,
    custom_css: str,
    rjsf_theme: str,
) -> None:
    """Add routes for FastAPI framework."""
    from fastapi.responses import HTMLResponse, JSONResponse
    from uf.templates import generate_index_html

    @app.get('/', response_class=HTMLResponse)
    async def index():
        """Serve main UI page."""
        try:
            html = generate_index_html(
                function_specs,
                page_title=page_title,
                custom_css=custom_css,
                rjsf_theme=rjsf_theme,
            )
            return html
        except Exception as e:
            from uf.templates import generate_error_page
            return HTMLResponse(
                content=generate_error_page(str(e), 500),
                status_code=500
            )

    @app.get('/api/functions')
    async def list_functions():
        """List all available functions."""
        try:
            func_list = [
                {
                    'name': name,
                    'description': spec.get('description', ''),
                }
                for name, spec in function_specs.items()
            ]
            return JSONResponse(content=func_list)
        except Exception as e:
            return JSONResponse(
                content={'error': str(e)},
                status_code=500
            )

    @app.get('/api/functions/{func_name}/spec')
    async def get_function_spec(func_name: str):
        """Get RJSF specification for a function."""
        try:
            spec = function_specs[func_name]
            return JSONResponse(content={
                'schema': spec['schema'],
                'uiSchema': spec['uiSchema'],
                'name': spec['name'],
                'description': spec['description'],
            })
        except KeyError:
            return JSONResponse(
                content={'error': f"Function '{func_name}' not found"},
                status_code=404
            )
        except Exception as e:
            return JSONResponse(
                content={'error': str(e)},
                status_code=500
            )


def create_function_handler(func: Callable, func_name: str) -> Callable:
    """Create a request handler for a function.

    This wraps the function to handle HTTP requests and responses.

    Args:
        func: The function to wrap
        func_name: Name of the function

    Returns:
        A handler function compatible with web frameworks
    """
    def handler(**kwargs):
        """Handle function execution from HTTP request."""
        try:
            result = func(**kwargs)
            return {'result': result, 'success': True}
        except Exception as e:
            return {
                'error': str(e),
                'success': False,
                'error_type': type(e).__name__,
            }

    handler.__name__ = func_name
    handler.__doc__ = func.__doc__
    return handler
