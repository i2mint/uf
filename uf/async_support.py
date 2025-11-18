"""Async function support for uf.

Provides utilities for detecting and handling async functions,
allowing seamless integration of async def functions in uf apps.
"""

import asyncio
import inspect
from typing import Callable, Any
from functools import wraps


def is_async_function(func: Callable) -> bool:
    """Check if a function is async.

    Args:
        func: Function to check

    Returns:
        True if function is async

    Example:
        >>> async def my_async_func():
        ...     pass
        >>> is_async_function(my_async_func)
        True
    """
    return asyncio.iscoroutinefunction(func)


def async_to_sync(async_func: Callable) -> Callable:
    """Convert an async function to a synchronous wrapper.

    Args:
        async_func: Async function to wrap

    Returns:
        Synchronous wrapper function

    Example:
        >>> async def fetch_data(url: str):
        ...     # async operations
        ...     return data
        >>> sync_fetch = async_to_sync(fetch_data)
        >>> result = sync_fetch('https://example.com')
    """
    if not is_async_function(async_func):
        return async_func

    @wraps(async_func)
    def sync_wrapper(*args, **kwargs):
        """Synchronous wrapper that runs async function."""
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Loop is already running, create a new one in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        async_func(*args, **kwargs)
                    )
                    return future.result()
            else:
                # Loop exists but not running
                return loop.run_until_complete(async_func(*args, **kwargs))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(async_func(*args, **kwargs))

    # Preserve metadata
    sync_wrapper.__uf_is_async__ = True
    sync_wrapper.__uf_original_async__ = async_func

    return sync_wrapper


def make_sync_compatible(funcs: list[Callable]) -> list[Callable]:
    """Convert any async functions in the list to sync wrappers.

    Args:
        funcs: List of functions (may include async)

    Returns:
        List of functions (all synchronous)

    Example:
        >>> funcs = [sync_func, async_func, another_sync]
        >>> compatible = make_sync_compatible(funcs)
        >>> # All functions in compatible are now callable synchronously
    """
    return [async_to_sync(func) for func in funcs]


def create_async_handler(async_func: Callable) -> Callable:
    """Create a handler for async functions in web frameworks.

    This is useful for integrating with frameworks that may or may not
    support async natively (like Bottle vs FastAPI).

    Args:
        async_func: Async function to create handler for

    Returns:
        Appropriate handler (async or sync wrapped)

    Example:
        >>> async def my_endpoint(param: str):
        ...     result = await some_async_operation(param)
        ...     return result
        >>> handler = create_async_handler(my_endpoint)
    """
    if not is_async_function(async_func):
        return async_func

    # For now, return sync wrapper
    # In future, could detect framework and return async if supported
    return async_to_sync(async_func)


class AsyncFunctionWrapper:
    """Wrapper for async functions that provides both sync and async access.

    This allows the same function to be called either way, depending on
    the context.

    Example:
        >>> async def fetch_user(user_id: int):
        ...     # async database query
        ...     return user_data
        >>>
        >>> wrapper = AsyncFunctionWrapper(fetch_user)
        >>> # Sync call
        >>> user = wrapper.call_sync(user_id=123)
        >>> # Async call
        >>> user = await wrapper.call_async(user_id=123)
    """

    def __init__(self, func: Callable):
        """Initialize the wrapper.

        Args:
            func: Function to wrap (sync or async)
        """
        self.func = func
        self.is_async = is_async_function(func)

    def call_sync(self, *args, **kwargs) -> Any:
        """Call the function synchronously.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        if self.is_async:
            return async_to_sync(self.func)(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)

    async def call_async(self, *args, **kwargs) -> Any:
        """Call the function asynchronously.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        if self.is_async:
            return await self.func(*args, **kwargs)
        else:
            # Run sync function in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.func(*args, **kwargs)
            )

    def __call__(self, *args, **kwargs):
        """Call synchronously by default.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        return self.call_sync(*args, **kwargs)


def batch_async_calls(
    async_func: Callable,
    calls: list[dict],
) -> list[Any]:
    """Execute multiple async function calls concurrently.

    Args:
        async_func: Async function to call
        calls: List of dicts with 'args' and 'kwargs' for each call

    Returns:
        List of results in same order as calls

    Example:
        >>> async def fetch_user(user_id: int):
        ...     return await db.get_user(user_id)
        >>>
        >>> calls = [
        ...     {'args': (), 'kwargs': {'user_id': 1}},
        ...     {'args': (), 'kwargs': {'user_id': 2}},
        ...     {'args': (), 'kwargs': {'user_id': 3}},
        ... ]
        >>> users = batch_async_calls(fetch_user, calls)
    """
    if not is_async_function(async_func):
        # Sync function, just call sequentially
        return [
            async_func(*call.get('args', ()), **call.get('kwargs', {}))
            for call in calls
        ]

    async def run_batch():
        tasks = [
            async_func(*call.get('args', ()), **call.get('kwargs', {}))
            for call in calls
        ]
        return await asyncio.gather(*tasks)

    return asyncio.run(run_batch())


def timeout_async(seconds: float):
    """Decorator to add timeout to async functions.

    Args:
        seconds: Timeout in seconds

    Returns:
        Decorator function

    Example:
        >>> @timeout_async(5.0)
        ... async def slow_operation():
        ...     await asyncio.sleep(10)  # Will timeout after 5s
    """

    def decorator(async_func: Callable) -> Callable:
        if not is_async_function(async_func):
            return async_func

        @wraps(async_func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    async_func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Function {async_func.__name__} "
                    f"timed out after {seconds} seconds"
                )

        return wrapper

    return decorator


def retry_async(max_attempts: int = 3, delay: float = 1.0):
    """Decorator to add retry logic to async functions.

    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds

    Returns:
        Decorator function

    Example:
        >>> @retry_async(max_attempts=3, delay=2.0)
        ... async def unreliable_api_call():
        ...     # May fail, will retry
        ...     return await external_api.fetch()
    """

    def decorator(async_func: Callable) -> Callable:
        if not is_async_function(async_func):
            return async_func

        @wraps(async_func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await async_func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay)
                    else:
                        raise last_exception

        return wrapper

    return decorator


class AsyncContext:
    """Context manager for handling async operations in uf.

    This helps manage event loops and async resources properly.

    Example:
        >>> with AsyncContext() as ctx:
        ...     result = ctx.run(my_async_function(param))
    """

    def __init__(self):
        """Initialize async context."""
        self._loop = None
        self._owned_loop = False

    def __enter__(self):
        """Enter the context."""
        try:
            self._loop = asyncio.get_event_loop()
            self._owned_loop = False
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._owned_loop = True

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        if self._owned_loop and self._loop:
            self._loop.close()

    def run(self, coro):
        """Run a coroutine in this context.

        Args:
            coro: Coroutine to run

        Returns:
            Result of the coroutine
        """
        if self._loop.is_running():
            # Can't run_until_complete on a running loop
            # Use asyncio.run in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return self._loop.run_until_complete(coro)


def detect_async_framework(app) -> str:
    """Detect if the web framework supports async natively.

    Args:
        app: The web application

    Returns:
        Framework name: 'fastapi', 'bottle', 'unknown'
    """
    app_type = type(app).__name__

    if 'FastAPI' in app_type:
        return 'fastapi'
    elif 'Bottle' in app_type or hasattr(app, 'route'):
        return 'bottle'
    else:
        return 'unknown'


def is_framework_async_capable(app) -> bool:
    """Check if the framework supports async handlers natively.

    Args:
        app: The web application

    Returns:
        True if framework supports async
    """
    framework = detect_async_framework(app)
    return framework in ['fastapi', 'aiohttp', 'starlette']
