"""Result caching for uf.

Provides caching decorators and backends to cache function results,
improving performance for expensive operations.
"""

from typing import Callable, Any, Optional, Hashable
from functools import wraps
from datetime import datetime, timedelta
import json
import hashlib
import pickle


class CacheBackend:
    """Base class for cache backends."""

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        """Delete a key from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all cache entries."""
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if exists and not expired
        """
        return self.get(key) is not None


class MemoryCache(CacheBackend):
    """In-memory cache backend.

    Simple dictionary-based caching suitable for single-process applications.

    Example:
        >>> cache = MemoryCache(default_ttl=3600)
        >>> cache.set('key', 'value', ttl=60)
        >>> value = cache.get('key')
    """

    def __init__(self, default_ttl: int = 3600, max_size: int = 1000):
        """Initialize memory cache.

        Args:
            default_ttl: Default TTL in seconds
            max_size: Maximum number of entries
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: dict[str, dict] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        entry = self._cache.get(key)
        if not entry:
            return None

        # Check expiration
        if entry['expires_at'] and datetime.now() > entry['expires_at']:
            del self._cache[key]
            return None

        return entry['value']

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache."""
        if ttl is None:
            ttl = self.default_ttl

        expires_at = None
        if ttl > 0:
            expires_at = datetime.now() + timedelta(seconds=ttl)

        self._cache[key] = {
            'value': value,
            'expires_at': expires_at,
            'created_at': datetime.now(),
        }

        # Evict oldest entries if over max size
        if len(self._cache) > self.max_size:
            self._evict_oldest()

    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def _evict_oldest(self) -> None:
        """Evict oldest entries to fit max size."""
        # Sort by created_at and remove oldest 10%
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: self._cache[k]['created_at']
        )

        num_to_remove = max(1, len(self._cache) // 10)
        for key in sorted_keys[:num_to_remove]:
            del self._cache[key]

    def cleanup_expired(self) -> int:
        """Remove expired entries.

        Returns:
            Number of entries removed
        """
        now = datetime.now()
        expired = [
            key for key, entry in self._cache.items()
            if entry['expires_at'] and now > entry['expires_at']
        ]

        for key in expired:
            del self._cache[key]

        return len(expired)

    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total = len(self._cache)
        now = datetime.now()
        expired = sum(
            1 for entry in self._cache.values()
            if entry['expires_at'] and now > entry['expires_at']
        )

        return {
            'total_entries': total,
            'active_entries': total - expired,
            'expired_entries': expired,
            'max_size': self.max_size,
            'utilization': total / self.max_size if self.max_size > 0 else 0,
        }


class DiskCache(CacheBackend):
    """Disk-based cache backend using pickle.

    Persists cache to disk, suitable for larger datasets or persistence
    across restarts.

    Example:
        >>> cache = DiskCache(cache_dir='/tmp/uf_cache')
        >>> cache.set('expensive_result', big_data)
    """

    def __init__(self, cache_dir: str = '.uf_cache', default_ttl: int = 3600):
        """Initialize disk cache.

        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default TTL in seconds
        """
        import os
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl

        os.makedirs(cache_dir, exist_ok=True)

    def _get_path(self, key: str) -> str:
        """Get file path for a key."""
        import os
        # Hash the key to create valid filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.cache")

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        import os

        path = self._get_path(key)
        if not os.path.exists(path):
            return None

        try:
            with open(path, 'rb') as f:
                entry = pickle.load(f)

            # Check expiration
            if entry['expires_at'] and datetime.now() > entry['expires_at']:
                os.remove(path)
                return None

            return entry['value']
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache."""
        if ttl is None:
            ttl = self.default_ttl

        expires_at = None
        if ttl > 0:
            expires_at = datetime.now() + timedelta(seconds=ttl)

        entry = {
            'value': value,
            'expires_at': expires_at,
            'created_at': datetime.now(),
        }

        path = self._get_path(key)
        with open(path, 'wb') as f:
            pickle.dump(entry, f)

    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        import os

        path = self._get_path(key)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        import os
        import glob

        for file_path in glob.glob(os.path.join(self.cache_dir, '*.cache')):
            os.remove(file_path)


def make_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Create a cache key from function call.

    Args:
        func_name: Function name
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Cache key string
    """
    # Create a deterministic key from arguments
    key_data = {
        'func': func_name,
        'args': args,
        'kwargs': sorted(kwargs.items()),
    }

    # Serialize to JSON for hashing
    try:
        key_str = json.dumps(key_data, sort_keys=True, default=str)
    except (TypeError, ValueError):
        # Fallback to string representation
        key_str = f"{func_name}:{args}:{sorted(kwargs.items())}"

    # Hash for compact key
    return hashlib.sha256(key_str.encode()).hexdigest()


def cached(
    ttl: int = 3600,
    backend: Optional[CacheBackend] = None,
    key_func: Optional[Callable] = None,
):
    """Decorator to cache function results.

    Args:
        ttl: Time to live in seconds
        backend: Cache backend (uses global MemoryCache if None)
        key_func: Optional function to generate cache key

    Returns:
        Decorator function

    Example:
        >>> @cached(ttl=3600)
        ... def expensive_calculation(x: int, y: int) -> int:
        ...     # Only runs once per unique (x, y) combination
        ...     return heavy_computation(x, y)
        >>>
        >>> result = expensive_calculation(10, 20)  # Computes
        >>> result2 = expensive_calculation(10, 20)  # From cache
    """
    if backend is None:
        backend = get_global_cache_backend()
        if backend is None:
            backend = MemoryCache(default_ttl=ttl)
            set_global_cache_backend(backend)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = make_cache_key(func.__name__, args, kwargs)

            # Try to get from cache
            cached_result = backend.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Compute result
            result = func(*args, **kwargs)

            # Store in cache
            backend.set(cache_key, result, ttl=ttl)

            return result

        wrapper.__uf_cached__ = True
        wrapper.__uf_cache_backend__ = backend
        wrapper.__uf_cache_ttl__ = ttl

        # Add cache control methods
        def clear_cache():
            """Clear all cached results for this function."""
            # This is a simplified version
            # Full implementation would track keys per function
            backend.clear()

        wrapper.clear_cache = clear_cache

        return wrapper

    return decorator


def cache_invalidate(cache_key: str, backend: Optional[CacheBackend] = None) -> bool:
    """Invalidate a specific cache entry.

    Args:
        cache_key: Cache key to invalidate
        backend: Cache backend (uses global if None)

    Returns:
        True if invalidated
    """
    if backend is None:
        backend = get_global_cache_backend()

    if backend:
        return backend.delete(cache_key)

    return False


def cache_clear_all(backend: Optional[CacheBackend] = None) -> None:
    """Clear all cache entries.

    Args:
        backend: Cache backend (uses global if None)
    """
    if backend is None:
        backend = get_global_cache_backend()

    if backend:
        backend.clear()


class CacheStats:
    """Track cache hit/miss statistics."""

    def __init__(self):
        """Initialize cache stats."""
        self.hits = 0
        self.misses = 0
        self.sets = 0

    def record_hit(self):
        """Record a cache hit."""
        self.hits += 1

    def record_miss(self):
        """Record a cache miss."""
        self.misses += 1

    def record_set(self):
        """Record a cache set."""
        self.sets += 1

    def hit_rate(self) -> float:
        """Calculate hit rate.

        Returns:
            Hit rate as a percentage (0-100)
        """
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100

    def reset(self):
        """Reset all statistics."""
        self.hits = 0
        self.misses = 0
        self.sets = 0

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Statistics dictionary
        """
        return {
            'hits': self.hits,
            'misses': self.misses,
            'sets': self.sets,
            'hit_rate': self.hit_rate(),
            'total_requests': self.hits + self.misses,
        }


# Global cache backend
_global_cache_backend: Optional[CacheBackend] = None


def set_global_cache_backend(backend: CacheBackend) -> None:
    """Set the global cache backend."""
    global _global_cache_backend
    _global_cache_backend = backend


def get_global_cache_backend() -> Optional[CacheBackend]:
    """Get the global cache backend."""
    return _global_cache_backend


# Initialize default global backend
set_global_cache_backend(MemoryCache())
