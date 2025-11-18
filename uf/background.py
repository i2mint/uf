"""Background task execution for uf.

Provides decorators and utilities for running tasks in the background,
with support for queues, scheduling, and progress tracking.
"""

from typing import Callable, Any, Optional
from functools import wraps
from datetime import datetime
from enum import Enum
import threading
import queue
import uuid


class TaskStatus(Enum):
    """Status of a background task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task:
    """Represents a background task.

    Attributes:
        task_id: Unique task identifier
        func_name: Name of the function
        args: Positional arguments
        kwargs: Keyword arguments
        status: Current task status
        result: Task result (if completed)
        error: Error message (if failed)
        created_at: When task was created
        started_at: When task started running
        completed_at: When task completed
        progress: Progress percentage (0-100)
    """

    def __init__(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        task_id: Optional[str] = None,
    ):
        """Initialize task.

        Args:
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            task_id: Optional task ID (generated if not provided)
        """
        self.task_id = task_id or str(uuid.uuid4())
        self.func = func
        self.func_name = func.__name__
        self.args = args
        self.kwargs = kwargs or {}
        self.status = TaskStatus.PENDING
        self.result: Any = None
        self.error: Optional[str] = None
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.progress = 0

    def execute(self) -> Any:
        """Execute the task.

        Returns:
            Task result

        Raises:
            Exception: If task execution fails
        """
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()

        try:
            self.result = self.func(*self.args, **self.kwargs)
            self.status = TaskStatus.COMPLETED
            self.progress = 100
            return self.result
        except Exception as e:
            self.status = TaskStatus.FAILED
            self.error = str(e)
            raise
        finally:
            self.completed_at = datetime.now()

    def to_dict(self) -> dict:
        """Convert task to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'task_id': self.task_id,
            'func_name': self.func_name,
            'status': self.status.value,
            'result': self.result if self.status == TaskStatus.COMPLETED else None,
            'error': self.error,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'progress': self.progress,
        }


class TaskQueue:
    """FIFO queue for background tasks.

    Example:
        >>> task_queue = TaskQueue(num_workers=2)
        >>> task_queue.start()
        >>> task_id = task_queue.submit(expensive_function, x=10, y=20)
        >>> status = task_queue.get_status(task_id)
        >>> result = task_queue.get_result(task_id)
    """

    def __init__(self, num_workers: int = 1, max_queue_size: int = 100):
        """Initialize task queue.

        Args:
            num_workers: Number of worker threads
            max_queue_size: Maximum queue size
        """
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._tasks: dict[str, Task] = {}
        self._workers: list[threading.Thread] = []
        self._running = False

    def start(self) -> None:
        """Start worker threads."""
        if self._running:
            return

        self._running = True

        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"TaskWorker-{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def stop(self, wait: bool = True) -> None:
        """Stop worker threads.

        Args:
            wait: Whether to wait for threads to finish
        """
        self._running = False

        if wait:
            for worker in self._workers:
                worker.join(timeout=5.0)

        self._workers.clear()

    def submit(
        self,
        func: Callable,
        *args,
        task_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Submit a task for execution.

        Args:
            func: Function to execute
            *args: Positional arguments
            task_id: Optional task ID
            **kwargs: Keyword arguments

        Returns:
            Task ID

        Raises:
            queue.Full: If queue is full
        """
        task = Task(func, args=args, kwargs=kwargs, task_id=task_id)
        self._tasks[task.task_id] = task
        self._queue.put(task, block=False)  # Don't block
        return task.task_id

    def get_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status.

        Args:
            task_id: Task ID

        Returns:
            TaskStatus or None if not found
        """
        task = self._tasks.get(task_id)
        return task.status if task else None

    def get_result(self, task_id: str, wait: bool = False, timeout: Optional[float] = None) -> Any:
        """Get task result.

        Args:
            task_id: Task ID
            wait: Whether to wait for completion
            timeout: Optional timeout in seconds

        Returns:
            Task result

        Raises:
            ValueError: If task not found
            RuntimeError: If task failed
            TimeoutError: If wait times out
        """
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        if wait and task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            # Wait for completion
            import time
            start_time = time.time()
            while task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                time.sleep(0.1)
                if timeout and (time.time() - start_time) > timeout:
                    raise TimeoutError(f"Task {task_id} timed out")

        if task.status == TaskStatus.FAILED:
            raise RuntimeError(f"Task failed: {task.error}")

        if task.status != TaskStatus.COMPLETED:
            return None

        return task.result

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task object.

        Args:
            task_id: Task ID

        Returns:
            Task object or None
        """
        return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.

        Args:
            task_id: Task ID

        Returns:
            True if cancelled

        Note:
            Can only cancel pending tasks
        """
        task = self._tasks.get(task_id)
        if not task or task.status != TaskStatus.PENDING:
            return False

        task.status = TaskStatus.CANCELLED
        return True

    def _worker_loop(self) -> None:
        """Worker thread loop."""
        while self._running:
            try:
                # Get task with timeout to allow checking _running
                task = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if task.status == TaskStatus.CANCELLED:
                continue

            try:
                task.execute()
            except Exception:
                # Error already recorded in task
                pass
            finally:
                self._queue.task_done()

    def queue_size(self) -> int:
        """Get current queue size.

        Returns:
            Number of pending tasks
        """
        return self._queue.qsize()

    def stats(self) -> dict:
        """Get queue statistics.

        Returns:
            Dictionary with statistics
        """
        total = len(self._tasks)
        by_status = {}
        for task in self._tasks.values():
            status = task.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return {
            'total_tasks': total,
            'queue_size': self.queue_size(),
            'num_workers': self.num_workers,
            'by_status': by_status,
        }


def background(
    queue_name: str = 'default',
    task_queue: Optional[TaskQueue] = None,
):
    """Decorator to run function in background.

    Args:
        queue_name: Name of the queue to use
        task_queue: Optional TaskQueue instance

    Returns:
        Decorator function

    Example:
        >>> @background()
        ... def send_email(to: str, subject: str):
        ...     # Long-running email sending
        ...     pass
        >>>
        >>> task_id = send_email('user@example.com', 'Hello')
        >>> # Returns immediately with task_id
    """
    if task_queue is None:
        task_queue = get_global_task_queue(queue_name)
        if task_queue is None:
            task_queue = TaskQueue(num_workers=2)
            task_queue.start()
            set_global_task_queue(queue_name, task_queue)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            """Submit task and return task ID."""
            task_id = task_queue.submit(func, *args, **kwargs)
            return task_id

        wrapper.__uf_background__ = True
        wrapper.__uf_task_queue__ = task_queue
        wrapper.__uf_original_func__ = func

        # Add utility methods
        def get_status(task_id: str):
            """Get task status."""
            return task_queue.get_status(task_id)

        def get_result(task_id: str, wait: bool = False, timeout: Optional[float] = None):
            """Get task result."""
            return task_queue.get_result(task_id, wait=wait, timeout=timeout)

        wrapper.get_status = get_status
        wrapper.get_result = get_result

        return wrapper

    return decorator


class PeriodicTask:
    """Task that runs periodically.

    Example:
        >>> def cleanup():
        ...     print("Cleaning up...")
        >>>
        >>> periodic = PeriodicTask(cleanup, interval=3600)
        >>> periodic.start()
        >>> # Runs every hour
        >>> periodic.stop()
    """

    def __init__(self, func: Callable, interval: float, args: tuple = (), kwargs: Optional[dict] = None):
        """Initialize periodic task.

        Args:
            func: Function to run
            interval: Interval in seconds
            args: Positional arguments
            kwargs: Keyword arguments
        """
        self.func = func
        self.interval = interval
        self.args = args
        self.kwargs = kwargs or {}
        self._timer: Optional[threading.Timer] = None
        self._running = False

    def start(self) -> None:
        """Start periodic execution."""
        if self._running:
            return

        self._running = True
        self._schedule_next()

    def stop(self) -> None:
        """Stop periodic execution."""
        self._running = False
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _schedule_next(self) -> None:
        """Schedule next execution."""
        if not self._running:
            return

        self._timer = threading.Timer(self.interval, self._run)
        self._timer.daemon = True
        self._timer.start()

    def _run(self) -> None:
        """Run the function and schedule next."""
        try:
            self.func(*self.args, **self.kwargs)
        except Exception:
            # Log error but continue
            pass
        finally:
            self._schedule_next()


# Global task queues
_global_task_queues: dict[str, TaskQueue] = {}


def get_global_task_queue(name: str = 'default') -> Optional[TaskQueue]:
    """Get a global task queue by name.

    Args:
        name: Queue name

    Returns:
        TaskQueue or None
    """
    return _global_task_queues.get(name)


def set_global_task_queue(name: str, task_queue: TaskQueue) -> None:
    """Set a global task queue.

    Args:
        name: Queue name
        task_queue: TaskQueue instance
    """
    _global_task_queues[name] = task_queue


def get_or_create_task_queue(name: str = 'default', num_workers: int = 2) -> TaskQueue:
    """Get or create a task queue.

    Args:
        name: Queue name
        num_workers: Number of workers if creating

    Returns:
        TaskQueue instance
    """
    queue = get_global_task_queue(name)
    if queue is None:
        queue = TaskQueue(num_workers=num_workers)
        queue.start()
        set_global_task_queue(name, queue)
    return queue
