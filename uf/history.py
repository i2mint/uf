"""Call history and presets for uf.

Provides functionality to track function calls, save parameter presets,
and reuse previous calls for improved user experience.
"""

from typing import Callable, Any, Optional
from datetime import datetime
from collections import defaultdict
import json


class FunctionCall:
    """Record of a single function call.

    Attributes:
        func_name: Name of the function
        params: Parameters used
        result: Result returned (if captured)
        timestamp: When the call was made
        success: Whether the call succeeded
        error: Error message if failed
    """

    def __init__(
        self,
        func_name: str,
        params: dict,
        result: Any = None,
        success: bool = True,
        error: Optional[str] = None,
    ):
        """Initialize function call record.

        Args:
            func_name: Name of the function
            params: Parameters dictionary
            result: Function result
            success: Whether call succeeded
            error: Error message if failed
        """
        self.func_name = func_name
        self.params = params
        self.result = result
        self.timestamp = datetime.now()
        self.success = success
        self.error = error

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'func_name': self.func_name,
            'params': self.params,
            'result': self.result if self.success else None,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'error': self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FunctionCall':
        """Create from dictionary.

        Args:
            data: Dictionary with call data

        Returns:
            FunctionCall instance
        """
        call = cls(
            func_name=data['func_name'],
            params=data['params'],
            result=data.get('result'),
            success=data.get('success', True),
            error=data.get('error'),
        )
        if 'timestamp' in data:
            call.timestamp = datetime.fromisoformat(data['timestamp'])
        return call


class CallHistory:
    """Manage history of function calls.

    Example:
        >>> history = CallHistory(max_size=100)
        >>> history.record('add', {'x': 10, 'y': 20}, result=30)
        >>> recent = history.get_recent('add', limit=5)
    """

    def __init__(self, max_size: int = 100):
        """Initialize call history.

        Args:
            max_size: Maximum number of calls to keep per function
        """
        self.max_size = max_size
        self._history: dict[str, list[FunctionCall]] = defaultdict(list)

    def record(
        self,
        func_name: str,
        params: dict,
        result: Any = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Record a function call.

        Args:
            func_name: Name of the function
            params: Parameters used
            result: Result returned
            success: Whether call succeeded
            error: Error message if failed
        """
        call = FunctionCall(
            func_name=func_name,
            params=params,
            result=result,
            success=success,
            error=error,
        )

        self._history[func_name].append(call)

        # Trim to max size
        if len(self._history[func_name]) > self.max_size:
            self._history[func_name] = self._history[func_name][-self.max_size:]

    def get_recent(self, func_name: str, limit: int = 10) -> list[FunctionCall]:
        """Get recent calls for a function.

        Args:
            func_name: Function name
            limit: Maximum number to return

        Returns:
            List of recent FunctionCall objects (newest first)
        """
        calls = self._history.get(func_name, [])
        return list(reversed(calls[-limit:]))

    def get_successful_calls(
        self, func_name: str, limit: int = 10
    ) -> list[FunctionCall]:
        """Get recent successful calls.

        Args:
            func_name: Function name
            limit: Maximum number to return

        Returns:
            List of successful FunctionCall objects
        """
        calls = [c for c in self._history.get(func_name, []) if c.success]
        return list(reversed(calls[-limit:]))

    def clear(self, func_name: Optional[str] = None) -> None:
        """Clear history.

        Args:
            func_name: Function to clear, or None for all
        """
        if func_name:
            self._history[func_name] = []
        else:
            self._history.clear()

    def to_dict(self) -> dict:
        """Convert history to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            func_name: [call.to_dict() for call in calls]
            for func_name, calls in self._history.items()
        }

    @classmethod
    def from_dict(cls, data: dict, max_size: int = 100) -> 'CallHistory':
        """Create from dictionary.

        Args:
            data: Dictionary with history data
            max_size: Maximum size per function

        Returns:
            CallHistory instance
        """
        history = cls(max_size=max_size)

        for func_name, calls_data in data.items():
            history._history[func_name] = [
                FunctionCall.from_dict(call_data) for call_data in calls_data
            ]

        return history


class Preset:
    """A saved parameter preset for a function.

    Attributes:
        name: Preset name
        func_name: Function this preset is for
        params: Parameter values
        description: Optional description
        created_at: When preset was created
    """

    def __init__(
        self,
        name: str,
        func_name: str,
        params: dict,
        description: str = "",
    ):
        """Initialize preset.

        Args:
            name: Preset name
            func_name: Function name
            params: Parameter values
            description: Optional description
        """
        self.name = name
        self.func_name = func_name
        self.params = params
        self.description = description
        self.created_at = datetime.now()

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'name': self.name,
            'func_name': self.func_name,
            'params': self.params,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Preset':
        """Create from dictionary.

        Args:
            data: Dictionary with preset data

        Returns:
            Preset instance
        """
        preset = cls(
            name=data['name'],
            func_name=data['func_name'],
            params=data['params'],
            description=data.get('description', ''),
        )
        if 'created_at' in data:
            preset.created_at = datetime.fromisoformat(data['created_at'])
        return preset


class PresetManager:
    """Manage parameter presets for functions.

    Example:
        >>> presets = PresetManager()
        >>> presets.save('quick_add', 'add', {'x': 10, 'y': 20}, 'Quick test')
        >>> preset = presets.get('quick_add', 'add')
        >>> result = my_func(**preset.params)
    """

    def __init__(self):
        """Initialize preset manager."""
        self._presets: dict[str, dict[str, Preset]] = defaultdict(dict)

    def save(
        self,
        preset_name: str,
        func_name: str,
        params: dict,
        description: str = "",
    ) -> Preset:
        """Save a parameter preset.

        Args:
            preset_name: Name for this preset
            func_name: Function this preset is for
            params: Parameter values
            description: Optional description

        Returns:
            Created Preset object
        """
        preset = Preset(
            name=preset_name,
            func_name=func_name,
            params=params,
            description=description,
        )

        self._presets[func_name][preset_name] = preset
        return preset

    def get(self, preset_name: str, func_name: str) -> Optional[Preset]:
        """Get a preset.

        Args:
            preset_name: Name of preset
            func_name: Function name

        Returns:
            Preset object or None
        """
        return self._presets.get(func_name, {}).get(preset_name)

    def list_presets(self, func_name: str) -> list[Preset]:
        """List all presets for a function.

        Args:
            func_name: Function name

        Returns:
            List of Preset objects
        """
        return list(self._presets.get(func_name, {}).values())

    def delete(self, preset_name: str, func_name: str) -> bool:
        """Delete a preset.

        Args:
            preset_name: Name of preset
            func_name: Function name

        Returns:
            True if deleted, False if not found
        """
        if func_name in self._presets and preset_name in self._presets[func_name]:
            del self._presets[func_name][preset_name]
            return True
        return False

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            func_name: {
                preset_name: preset.to_dict()
                for preset_name, preset in presets.items()
            }
            for func_name, presets in self._presets.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PresetManager':
        """Create from dictionary.

        Args:
            data: Dictionary with preset data

        Returns:
            PresetManager instance
        """
        manager = cls()

        for func_name, presets_data in data.items():
            for preset_name, preset_data in presets_data.items():
                preset = Preset.from_dict(preset_data)
                manager._presets[func_name][preset_name] = preset

        return manager


class HistoryManager:
    """Combined manager for history and presets.

    Provides a unified interface for tracking calls and managing presets.

    Example:
        >>> manager = HistoryManager()
        >>> manager.record_call('add', {'x': 10, 'y': 20}, result=30)
        >>> manager.save_preset('quick', 'add', {'x': 10, 'y': 20})
        >>> recent = manager.get_recent_calls('add')
        >>> presets = manager.get_presets('add')
    """

    def __init__(self, max_history: int = 100):
        """Initialize history manager.

        Args:
            max_history: Maximum history size per function
        """
        self.history = CallHistory(max_size=max_history)
        self.presets = PresetManager()

    def record_call(
        self,
        func_name: str,
        params: dict,
        result: Any = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Record a function call.

        Args:
            func_name: Function name
            params: Parameters used
            result: Result returned
            success: Whether call succeeded
            error: Error message if failed
        """
        self.history.record(func_name, params, result, success, error)

    def get_recent_calls(
        self, func_name: str, limit: int = 10
    ) -> list[FunctionCall]:
        """Get recent calls.

        Args:
            func_name: Function name
            limit: Maximum number to return

        Returns:
            List of recent FunctionCall objects
        """
        return self.history.get_recent(func_name, limit)

    def save_preset(
        self,
        preset_name: str,
        func_name: str,
        params: dict,
        description: str = "",
    ) -> Preset:
        """Save a preset.

        Args:
            preset_name: Name for preset
            func_name: Function name
            params: Parameter values
            description: Optional description

        Returns:
            Created Preset object
        """
        return self.presets.save(preset_name, func_name, params, description)

    def get_preset(self, preset_name: str, func_name: str) -> Optional[Preset]:
        """Get a preset.

        Args:
            preset_name: Name of preset
            func_name: Function name

        Returns:
            Preset object or None
        """
        return self.presets.get(preset_name, func_name)

    def get_presets(self, func_name: str) -> list[Preset]:
        """Get all presets for a function.

        Args:
            func_name: Function name

        Returns:
            List of Preset objects
        """
        return self.presets.list_presets(func_name)

    def save_to_file(self, filepath: str) -> None:
        """Save history and presets to file.

        Args:
            filepath: Path to save to
        """
        data = {
            'history': self.history.to_dict(),
            'presets': self.presets.to_dict(),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str, max_history: int = 100) -> 'HistoryManager':
        """Load from file.

        Args:
            filepath: Path to load from
            max_history: Maximum history size

        Returns:
            HistoryManager instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        manager = cls(max_history=max_history)
        manager.history = CallHistory.from_dict(data.get('history', {}), max_history)
        manager.presets = PresetManager.from_dict(data.get('presets', {}))

        return manager


# Global instance
_global_history_manager = HistoryManager()


def get_global_history_manager() -> HistoryManager:
    """Get the global history manager.

    Returns:
        Global HistoryManager instance
    """
    return _global_history_manager


def enable_history(func: Callable, max_size: int = 100) -> Callable:
    """Decorator to enable call history for a function.

    Args:
        max_size: Maximum history size

    Returns:
        Decorator function

    Example:
        >>> @enable_history
        ... def my_function(x: int):
        ...     return x * 2
    """

    def wrapper(*args, **kwargs):
        """Wrapper that records calls."""
        manager = get_global_history_manager()

        try:
            result = func(*args, **kwargs)
            manager.record_call(
                func.__name__,
                kwargs,
                result=result,
                success=True,
            )
            return result
        except Exception as e:
            manager.record_call(
                func.__name__,
                kwargs,
                success=False,
                error=str(e),
            )
            raise

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__uf_history_enabled__ = True

    return wrapper
