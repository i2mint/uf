# uf.caching

Result caching for uf.

Provides caching decorators and backends to cache function results,
improving performance for expensive operations.

### Functions

| [`cache_clear_all`](#uf.caching.cache_clear_all)([backend])              | Clear all cache entries.               |
|------------------------------------------------------------------------------------------|----------------------------------------|
| [`cache_invalidate`](#uf.caching.cache_invalidate)(cache_key[, backend])  | Invalidate a specific cache entry.     |
| [`cached`](#uf.caching.cached)([ttl, backend, key_func])        | Decorator to cache function results.   |
| [`get_global_cache_backend`](#uf.caching.get_global_cache_backend)()              | Get the global cache backend.          |
| [`make_cache_key`](#uf.caching.make_cache_key)(func_name, args, kwargs) | Create a cache key from function call. |
| [`set_global_cache_backend`](#uf.caching.set_global_cache_backend)(backend)       | Set the global cache backend.          |

### Classes

| [`CacheBackend`](#uf.caching.CacheBackend)()                       | Base class for cache backends.         |
|---------------------------------------------------------------------------------------|----------------------------------------|
| [`CacheStats`](#uf.caching.CacheStats)()                         | Track cache hit/miss statistics.       |
| [`DiskCache`](#uf.caching.DiskCache)([cache_dir, default_ttl])  | Disk-based cache backend using pickle. |
| [`MemoryCache`](#uf.caching.MemoryCache)([default_ttl, max_size]) | In-memory cache backend.               |

### *class* uf.caching.CacheBackend

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Base class for cache backends.

#### clear()

Clear all cache entries.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### delete(key)

Delete a key from cache.

* **Parameters:**
  **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Cache key
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if deleted

#### exists(key)

Check if key exists in cache.

* **Parameters:**
  **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Cache key
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if exists and not expired

#### get(key)

Get a value from cache.

* **Parameters:**
  **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Cache key
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
* **Returns:**
  Cached value or None if not found/expired

#### set(key, value, ttl=None)

Set a value in cache.

* **Parameters:**
  * **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Cache key
  * **value** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Value to cache
  * **ttl** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Time to live in seconds
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### *class* uf.caching.CacheStats

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Track cache hit/miss statistics.

#### hit_rate()

Calculate hit rate.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
* **Returns:**
  Hit rate as a percentage (0-100)

#### record_hit()

Record a cache hit.

#### record_miss()

Record a cache miss.

#### record_set()

Record a cache set.

#### reset()

Reset all statistics.

#### to_dict()

Convert to dictionary.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Statistics dictionary

### *class* uf.caching.DiskCache(cache_dir='.uf_cache', default_ttl=3600)

Bases: [`CacheBackend`](#uf.caching.CacheBackend)

Disk-based cache backend using pickle.

Persists cache to disk, suitable for larger datasets or persistence
across restarts.

### Example

```pycon
>>> cache = DiskCache(cache_dir='/tmp/uf_cache')
>>> cache.set('expensive_result', big_data)
```

#### clear()

Clear all cache entries.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### delete(key)

Delete a key from cache.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### get(key)

Get a value from cache.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

#### set(key, value, ttl=None)

Set a value in cache.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### *class* uf.caching.MemoryCache(default_ttl=3600, max_size=1000)

Bases: [`CacheBackend`](#uf.caching.CacheBackend)

In-memory cache backend.

Simple dictionary-based caching suitable for single-process applications.

### Example

```pycon
>>> cache = MemoryCache(default_ttl=3600)
>>> cache.set('key', 'value', ttl=60)
>>> value = cache.get('key')
```

#### cleanup_expired()

Remove expired entries.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)
* **Returns:**
  Number of entries removed

#### clear()

Clear all cache entries.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### delete(key)

Delete a key from cache.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### get(key)

Get a value from cache.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

#### set(key, value, ttl=None)

Set a value in cache.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### stats()

Get cache statistics.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary with cache stats

### uf.caching.cache_clear_all(backend=None)

Clear all cache entries.

* **Parameters:**
  **backend** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`CacheBackend`](#uf.caching.CacheBackend)]) – Cache backend (uses global if None)
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### uf.caching.cache_invalidate(cache_key, backend=None)

Invalidate a specific cache entry.

* **Parameters:**
  * **cache_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Cache key to invalidate
  * **backend** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`CacheBackend`](#uf.caching.CacheBackend)]) – Cache backend (uses global if None)
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if invalidated

### uf.caching.cached(ttl=3600, backend=None, key_func=None)

Decorator to cache function results.

* **Parameters:**
  * **ttl** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Time to live in seconds
  * **backend** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`CacheBackend`](#uf.caching.CacheBackend)]) – Cache backend (uses global MemoryCache if None)
  * **key_func** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Optional function to generate cache key
* **Returns:**
  Decorator function

### Example

```pycon
>>> @cached(ttl=3600)
... def expensive_calculation(x: int, y: int) -> int:
...     # Only runs once per unique (x, y) combination
...     return heavy_computation(x, y)
>>>
>>> result = expensive_calculation(10, 20)  # Computes
>>> result2 = expensive_calculation(10, 20)  # From cache
```

### uf.caching.get_global_cache_backend()

Get the global cache backend.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`CacheBackend`](#uf.caching.CacheBackend)]

### uf.caching.make_cache_key(func_name, args, kwargs)

Create a cache key from function call.

* **Parameters:**
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
  * **args** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)) – Positional arguments
  * **kwargs** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Keyword arguments
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Cache key string

### uf.caching.set_global_cache_backend(backend)

Set the global cache backend.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)
