"""OpenAPI/Swagger integration for uf.

Automatically generates OpenAPI specifications and provides Swagger UI
for API documentation and testing.
"""

from typing import Callable, Any, Optional
import inspect


def function_to_openapi_operation(func: Callable, path: str = None) -> dict:
    """Convert a function to OpenAPI operation spec.

    Args:
        func: Function to convert
        path: Optional API path

    Returns:
        OpenAPI operation dictionary
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""

    # Parse docstring for description
    lines = doc.split('\n')
    summary = lines[0] if lines else func.__name__
    description = '\n'.join(lines[1:]).strip() if len(lines) > 1 else summary

    # Build parameters from signature
    parameters = []
    request_body = None

    type_map = {
        int: 'integer',
        float: 'number',
        str: 'string',
        bool: 'boolean',
        list: 'array',
        dict: 'object',
    }

    for param_name, param in sig.parameters.items():
        param_type = 'string'  # default

        if param.annotation != inspect.Parameter.empty:
            py_type = param.annotation
            # Handle Optional types
            if hasattr(py_type, '__origin__'):
                if py_type.__origin__ is type(Optional):
                    py_type = py_type.__args__[0]

            param_type = type_map.get(py_type, 'string')

        param_schema = {
            'type': param_type,
        }

        # Check if required
        required = param.default == inspect.Parameter.empty

        parameters.append({
            'name': param_name,
            'in': 'query',
            'required': required,
            'schema': param_schema,
        })

    operation = {
        'summary': summary,
        'description': description,
        'parameters': parameters,
        'responses': {
            '200': {
                'description': 'Successful response',
                'content': {
                    'application/json': {
                        'schema': {'type': 'object'}
                    }
                }
            },
            '400': {
                'description': 'Bad request'
            },
            '500': {
                'description': 'Internal server error'
            }
        }
    }

    # Add tags if function has group
    if hasattr(func, '__uf_ui_config__'):
        config = func.__uf_ui_config__
        if config.get('group'):
            operation['tags'] = [config['group']]

    return operation


def generate_openapi_spec(
    funcs: list[Callable],
    title: str = "API",
    version: str = "1.0.0",
    description: str = "",
    servers: Optional[list[dict]] = None,
) -> dict:
    """Generate OpenAPI 3.0 specification.

    Args:
        funcs: List of functions
        title: API title
        version: API version
        description: API description
        servers: Optional list of server configs

    Returns:
        OpenAPI specification dictionary
    """
    if servers is None:
        servers = [{'url': '/'}]

    paths = {}
    tags = set()

    for func in funcs:
        func_name = func.__name__
        path = f'/{func_name}'

        operation = function_to_openapi_operation(func, path)

        # Collect tags
        if 'tags' in operation:
            tags.update(operation['tags'])

        paths[path] = {
            'post': operation  # Use POST for form submissions
        }

    spec = {
        'openapi': '3.0.0',
        'info': {
            'title': title,
            'version': version,
            'description': description,
        },
        'servers': servers,
        'paths': paths,
    }

    # Add tags
    if tags:
        spec['tags'] = [{'name': tag} for tag in sorted(tags)]

    return spec


def swagger_ui_html(openapi_url: str = '/openapi.json') -> str:
    """Generate Swagger UI HTML.

    Args:
        openapi_url: URL to OpenAPI spec

    Returns:
        HTML string for Swagger UI
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API Documentation</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {{
            window.ui = SwaggerUIBundle({{
                url: '{openapi_url}',
                dom_id: '#swagger-ui',
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                layout: "StandaloneLayout"
            }});
        }};
    </script>
</body>
</html>"""
    return html


def redoc_ui_html(openapi_url: str = '/openapi.json') -> str:
    """Generate ReDoc UI HTML.

    Args:
        openapi_url: URL to OpenAPI spec

    Returns:
        HTML string for ReDoc UI
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API Documentation</title>
</head>
<body>
    <redoc spec-url='{openapi_url}'></redoc>
    <script src="https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js"></script>
</body>
</html>"""
    return html


def add_openapi_routes(app: Any, funcs: list[Callable], **spec_kwargs) -> None:
    """Add OpenAPI routes to an app.

    Adds:
    - /openapi.json: OpenAPI specification
    - /docs: Swagger UI
    - /redoc: ReDoc UI

    Args:
        app: Web application
        funcs: List of functions
        **spec_kwargs: Arguments for generate_openapi_spec
    """
    # Generate spec
    spec = generate_openapi_spec(funcs, **spec_kwargs)

    # Detect framework
    is_bottle = hasattr(app, 'route')

    if is_bottle:
        @app.route('/openapi.json')
        def openapi_spec():
            """Return OpenAPI spec."""
            import json
            from bottle import response
            response.content_type = 'application/json'
            return json.dumps(spec)

        @app.route('/docs')
        def swagger_ui():
            """Return Swagger UI."""
            return swagger_ui_html()

        @app.route('/redoc')
        def redoc_ui():
            """Return ReDoc UI."""
            return redoc_ui_html()
    else:
        # FastAPI
        from fastapi.responses import JSONResponse, HTMLResponse

        @app.get('/openapi.json')
        async def openapi_spec():
            """Return OpenAPI spec."""
            return JSONResponse(content=spec)

        @app.get('/docs', response_class=HTMLResponse)
        async def swagger_ui():
            """Return Swagger UI."""
            return swagger_ui_html()

        @app.get('/redoc', response_class=HTMLResponse)
        async def redoc_ui():
            """Return ReDoc UI."""
            return redoc_ui_html()


class OpenAPIConfig:
    """Configuration for OpenAPI generation.

    Example:
        >>> config = OpenAPIConfig(
        ...     title="My API",
        ...     version="2.0.0",
        ...     description="API for my application"
        ... )
    """

    def __init__(
        self,
        title: str = "API",
        version: str = "1.0.0",
        description: str = "",
        servers: Optional[list[dict]] = None,
        enable_swagger: bool = True,
        enable_redoc: bool = True,
    ):
        """Initialize OpenAPI config.

        Args:
            title: API title
            version: API version
            description: API description
            servers: List of server configurations
            enable_swagger: Enable Swagger UI
            enable_redoc: Enable ReDoc UI
        """
        self.title = title
        self.version = version
        self.description = description
        self.servers = servers or [{'url': '/'}]
        self.enable_swagger = enable_swagger
        self.enable_redoc = enable_redoc

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Configuration dictionary
        """
        return {
            'title': self.title,
            'version': self.version,
            'description': self.description,
            'servers': self.servers,
            'enable_swagger': self.enable_swagger,
            'enable_redoc': self.enable_redoc,
        }
