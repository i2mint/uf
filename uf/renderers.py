"""Result rendering system for uf.

Provides smart rendering of function results based on their type,
including tables, charts, DataFrames, JSON, and custom renderers.
"""

from typing import Callable, Any, Optional, Type
from collections.abc import Mapping
import json


class ResultRenderer:
    """Base class for result renderers.

    Subclass this to create custom renderers for specific result types.
    """

    def can_render(self, result: Any) -> bool:
        """Check if this renderer can handle the given result.

        Args:
            result: The function result to render

        Returns:
            True if this renderer can handle the result
        """
        raise NotImplementedError

    def render(self, result: Any) -> dict:
        """Render the result to a displayable format.

        Args:
            result: The function result to render

        Returns:
            Dictionary with rendering information:
            - 'type': Renderer type (e.g., 'table', 'chart', 'json')
            - 'data': Rendered data
            - 'options': Optional rendering options
        """
        raise NotImplementedError


class JsonRenderer(ResultRenderer):
    """Render results as formatted JSON."""

    def can_render(self, result: Any) -> bool:
        """Can render any JSON-serializable result."""
        try:
            json.dumps(result)
            return True
        except (TypeError, ValueError):
            return False

    def render(self, result: Any) -> dict:
        """Render as formatted JSON."""
        return {
            'type': 'json',
            'data': result,
            'options': {'indent': 2},
        }


class TableRenderer(ResultRenderer):
    """Render list of dicts as a table."""

    def can_render(self, result: Any) -> bool:
        """Can render list of dictionaries."""
        if not isinstance(result, list):
            return False
        if not result:
            return False
        return all(isinstance(item, dict) for item in result)

    def render(self, result: Any) -> dict:
        """Render as table."""
        if not result:
            return {'type': 'table', 'data': [], 'columns': []}

        # Extract columns from first item
        columns = list(result[0].keys())

        return {
            'type': 'table',
            'data': result,
            'columns': columns,
            'options': {'sortable': True, 'searchable': True},
        }


class DataFrameRenderer(ResultRenderer):
    """Render pandas DataFrame."""

    def can_render(self, result: Any) -> bool:
        """Can render pandas DataFrame."""
        try:
            import pandas as pd
            return isinstance(result, pd.DataFrame)
        except ImportError:
            return False

    def render(self, result: Any) -> dict:
        """Render DataFrame as table."""
        # Convert to dict records
        data = result.to_dict('records')
        columns = result.columns.tolist()

        return {
            'type': 'dataframe',
            'data': data,
            'columns': columns,
            'options': {
                'sortable': True,
                'searchable': True,
                'index': result.index.tolist(),
            },
        }


class ChartRenderer(ResultRenderer):
    """Render data suitable for charts."""

    def can_render(self, result: Any) -> bool:
        """Can render list of dicts with numeric values."""
        if not isinstance(result, list):
            return False
        if not result:
            return False
        if not all(isinstance(item, dict) for item in result):
            return False

        # Check if has numeric values
        first = result[0]
        has_numeric = any(
            isinstance(v, (int, float)) for v in first.values()
        )
        return has_numeric

    def render(self, result: Any) -> dict:
        """Render as chart data."""
        if not result:
            return {'type': 'chart', 'data': []}

        # Extract labels and datasets
        first = result[0]
        label_key = list(first.keys())[0]  # First key is label
        value_keys = [k for k in first.keys() if isinstance(first[k], (int, float))]

        labels = [item[label_key] for item in result]
        datasets = []

        for value_key in value_keys:
            datasets.append({
                'label': value_key,
                'data': [item[value_key] for item in result],
            })

        return {
            'type': 'chart',
            'data': {
                'labels': labels,
                'datasets': datasets,
            },
            'options': {
                'chart_type': 'bar',  # default, can be overridden
                'responsive': True,
            },
        }


class ImageRenderer(ResultRenderer):
    """Render image data."""

    def can_render(self, result: Any) -> bool:
        """Can render bytes that look like images."""
        if isinstance(result, bytes):
            # Check for common image headers
            if result.startswith(b'\x89PNG'):
                return True
            if result.startswith(b'\xff\xd8\xff'):  # JPEG
                return True
            if result.startswith(b'GIF8'):
                return True
        return False

    def render(self, result: Any) -> dict:
        """Render image as base64."""
        import base64

        b64_data = base64.b64encode(result).decode('utf-8')

        # Detect format
        if result.startswith(b'\x89PNG'):
            mime_type = 'image/png'
        elif result.startswith(b'\xff\xd8\xff'):
            mime_type = 'image/jpeg'
        elif result.startswith(b'GIF8'):
            mime_type = 'image/gif'
        else:
            mime_type = 'image/png'

        return {
            'type': 'image',
            'data': f'data:{mime_type};base64,{b64_data}',
            'options': {},
        }


class ResultRendererRegistry:
    """Registry for result renderers.

    Manages a collection of renderers and selects the appropriate one
    for each result type.
    """

    def __init__(self):
        """Initialize the registry with default renderers."""
        self._renderers: list[ResultRenderer] = []
        self._type_renderers: dict[str, ResultRenderer] = {}

        # Register default renderers in priority order
        self.register(DataFrameRenderer())
        self.register(ImageRenderer())
        self.register(TableRenderer())
        self.register(ChartRenderer())
        self.register(JsonRenderer())  # Fallback

    def register(
        self,
        renderer: ResultRenderer,
        priority: int = 0,
    ) -> None:
        """Register a renderer.

        Args:
            renderer: ResultRenderer instance
            priority: Higher priority renderers are tried first
        """
        self._renderers.insert(priority, renderer)

    def register_for_type(
        self,
        result_type: Type,
        renderer: ResultRenderer,
    ) -> None:
        """Register a renderer for a specific type.

        Args:
            result_type: Python type to match
            renderer: ResultRenderer instance
        """
        type_name = result_type.__name__
        self._type_renderers[type_name] = renderer

    def render(self, result: Any, renderer_type: Optional[str] = None) -> dict:
        """Render a result using the appropriate renderer.

        Args:
            result: The function result to render
            renderer_type: Optional specific renderer type to use

        Returns:
            Rendered result dictionary
        """
        # If specific renderer requested, try that first
        if renderer_type and renderer_type in self._type_renderers:
            renderer = self._type_renderers[renderer_type]
            if renderer.can_render(result):
                return renderer.render(result)

        # Check type-specific renderers
        result_type_name = type(result).__name__
        if result_type_name in self._type_renderers:
            renderer = self._type_renderers[result_type_name]
            if renderer.can_render(result):
                return renderer.render(result)

        # Try each registered renderer
        for renderer in self._renderers:
            if renderer.can_render(result):
                return renderer.render(result)

        # Fallback to JSON
        return JsonRenderer().render(str(result))


# Global registry instance
_global_registry = ResultRendererRegistry()


def get_global_renderer_registry() -> ResultRendererRegistry:
    """Get the global renderer registry.

    Returns:
        The global ResultRendererRegistry instance
    """
    return _global_registry


def register_renderer(renderer: ResultRenderer, priority: int = 0) -> None:
    """Register a renderer in the global registry.

    Args:
        renderer: ResultRenderer instance
        priority: Higher priority renderers are tried first
    """
    _global_registry.register(renderer, priority)


def register_renderer_for_type(result_type: Type, renderer: ResultRenderer) -> None:
    """Register a renderer for a specific type.

    Args:
        result_type: Python type to match
        renderer: ResultRenderer instance
    """
    _global_registry.register_for_type(result_type, renderer)


def render_result(result: Any, renderer_type: Optional[str] = None) -> dict:
    """Render a result using the global registry.

    Args:
        result: The function result to render
        renderer_type: Optional specific renderer type to use

    Returns:
        Rendered result dictionary
    """
    return _global_registry.render(result, renderer_type)


def result_renderer(renderer_type: str):
    """Decorator to specify result renderer for a function.

    Args:
        renderer_type: Type of renderer to use

    Returns:
        Decorator function

    Example:
        >>> @result_renderer('table')
        ... def get_users() -> list[dict]:
        ...     return [{'name': 'Alice', 'age': 30}]
    """

    def decorator(func: Callable) -> Callable:
        setattr(func, '__uf_result_renderer__', renderer_type)
        return func

    return decorator


def get_result_renderer(func: Callable) -> Optional[str]:
    """Get the result renderer type for a function.

    Args:
        func: Function to check

    Returns:
        Renderer type string or None
    """
    return getattr(func, '__uf_result_renderer__', None)


# Custom renderer examples


class MarkdownRenderer(ResultRenderer):
    """Render markdown strings."""

    def can_render(self, result: Any) -> bool:
        """Can render strings that look like markdown."""
        if not isinstance(result, str):
            return False
        # Simple heuristic: contains markdown-like syntax
        md_indicators = ['#', '**', '*', '```', '[', '|']
        return any(indicator in result for indicator in md_indicators)

    def render(self, result: Any) -> dict:
        """Render as markdown."""
        return {
            'type': 'markdown',
            'data': result,
            'options': {},
        }


class HtmlRenderer(ResultRenderer):
    """Render HTML strings."""

    def can_render(self, result: Any) -> bool:
        """Can render strings that look like HTML."""
        if not isinstance(result, str):
            return False
        return result.strip().startswith('<') and '>' in result

    def render(self, result: Any) -> dict:
        """Render as HTML."""
        return {
            'type': 'html',
            'data': result,
            'options': {'sanitize': True},  # Security: sanitize HTML
        }


# Register additional renderers
register_renderer(MarkdownRenderer())
register_renderer(HtmlRenderer())
