"""Webhook support for uf.

Provides functionality to trigger HTTP callbacks when functions complete,
enabling integration with external services.
"""

from typing import Callable, Optional, Any
from functools import wraps
from datetime import datetime
import threading
import json


class WebhookEvent:
    """Represents a webhook event.

    Attributes:
        event_type: Type of event ('success', 'failure', 'start')
        func_name: Name of the function
        params: Function parameters
        result: Function result (if success)
        error: Error message (if failure)
        timestamp: When event occurred
    """

    def __init__(
        self,
        event_type: str,
        func_name: str,
        params: dict,
        result: Any = None,
        error: Optional[str] = None,
    ):
        """Initialize webhook event.

        Args:
            event_type: Event type
            func_name: Function name
            params: Function parameters
            result: Function result
            error: Error message
        """
        self.event_type = event_type
        self.func_name = func_name
        self.params = params
        self.result = result
        self.error = error
        self.timestamp = datetime.now()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation
        """
        return {
            'event_type': self.event_type,
            'func_name': self.func_name,
            'params': self.params,
            'result': self.result,
            'error': self.error,
            'timestamp': self.timestamp.isoformat(),
        }


class WebhookClient:
    """Client for sending webhooks.

    Example:
        >>> client = WebhookClient('https://example.com/webhook')
        >>> client.send(WebhookEvent('success', 'my_func', {}, result=42))
    """

    def __init__(
        self,
        url: str,
        headers: Optional[dict] = None,
        timeout: float = 10.0,
        retry_count: int = 3,
    ):
        """Initialize webhook client.

        Args:
            url: Webhook URL
            headers: Optional HTTP headers
            timeout: Request timeout in seconds
            retry_count: Number of retry attempts
        """
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self.retry_count = retry_count

    def send(self, event: WebhookEvent, async_send: bool = True) -> bool:
        """Send a webhook event.

        Args:
            event: WebhookEvent to send
            async_send: Whether to send asynchronously

        Returns:
            True if sent successfully (for sync sends)
        """
        if async_send:
            # Send in background thread
            thread = threading.Thread(
                target=self._send_sync,
                args=(event,),
                daemon=True,
            )
            thread.start()
            return True
        else:
            return self._send_sync(event)

    def _send_sync(self, event: WebhookEvent) -> bool:
        """Send webhook synchronously.

        Args:
            event: WebhookEvent to send

        Returns:
            True if sent successfully
        """
        import requests

        payload = event.to_dict()
        headers = {
            **self.headers,
            'Content-Type': 'application/json',
        }

        for attempt in range(self.retry_count):
            try:
                response = requests.post(
                    self.url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return True
            except Exception as e:
                if attempt == self.retry_count - 1:
                    # Last attempt failed
                    print(f"Webhook send failed after {self.retry_count} attempts: {e}")
                    return False
                # Retry with exponential backoff
                import time
                time.sleep(2 ** attempt)

        return False


class WebhookManager:
    """Manage webhooks for multiple URLs and events.

    Example:
        >>> manager = WebhookManager()
        >>> manager.add_webhook('https://example.com/hook1', events=['success'])
        >>> manager.add_webhook('https://example.com/hook2', events=['failure'])
        >>> manager.trigger('success', 'my_func', {}, result=42)
    """

    def __init__(self):
        """Initialize webhook manager."""
        self._webhooks: list[dict] = []

    def add_webhook(
        self,
        url: str,
        events: Optional[list[str]] = None,
        headers: Optional[dict] = None,
        condition: Optional[Callable] = None,
    ) -> None:
        """Add a webhook.

        Args:
            url: Webhook URL
            events: List of event types to trigger on (None = all)
            headers: Optional HTTP headers
            condition: Optional callable to filter events
        """
        self._webhooks.append({
            'url': url,
            'events': events,
            'headers': headers or {},
            'condition': condition,
            'client': WebhookClient(url, headers),
        })

    def remove_webhook(self, url: str) -> bool:
        """Remove a webhook by URL.

        Args:
            url: Webhook URL

        Returns:
            True if removed
        """
        original_len = len(self._webhooks)
        self._webhooks = [w for w in self._webhooks if w['url'] != url]
        return len(self._webhooks) < original_len

    def trigger(
        self,
        event_type: str,
        func_name: str,
        params: dict,
        result: Any = None,
        error: Optional[str] = None,
    ) -> int:
        """Trigger webhooks for an event.

        Args:
            event_type: Event type
            func_name: Function name
            params: Function parameters
            result: Function result
            error: Error message

        Returns:
            Number of webhooks triggered
        """
        event = WebhookEvent(event_type, func_name, params, result, error)

        triggered = 0
        for webhook in self._webhooks:
            # Check if this webhook should be triggered
            if webhook['events'] and event_type not in webhook['events']:
                continue

            # Check condition if provided
            if webhook['condition'] and not webhook['condition'](event):
                continue

            # Send webhook
            webhook['client'].send(event)
            triggered += 1

        return triggered

    def list_webhooks(self) -> list[dict]:
        """List all registered webhooks.

        Returns:
            List of webhook configurations
        """
        return [
            {
                'url': w['url'],
                'events': w['events'],
            }
            for w in self._webhooks
        ]


def webhook(
    on: Optional[list[str]] = None,
    url: Optional[str] = None,
    manager: Optional[WebhookManager] = None,
):
    """Decorator to add webhooks to a function.

    Args:
        on: List of events to trigger on ('success', 'failure', 'start')
        url: Optional webhook URL
        manager: Optional WebhookManager instance

    Returns:
        Decorator function

    Example:
        >>> @webhook(on=['success', 'failure'])
        ... def process_order(order_id: int):
        ...     # Process order
        ...     return {'status': 'processed'}
    """
    if on is None:
        on = ['success', 'failure']

    if manager is None:
        manager = get_global_webhook_manager()
        if manager is None:
            manager = WebhookManager()
            set_global_webhook_manager(manager)

    # If URL provided, add to manager
    if url:
        manager.add_webhook(url, events=on)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get params from kwargs
            params = kwargs.copy()

            # Trigger start event if configured
            if 'start' in on:
                manager.trigger('start', func.__name__, params)

            try:
                result = func(*args, **kwargs)

                # Trigger success event
                if 'success' in on:
                    manager.trigger('success', func.__name__, params, result=result)

                return result

            except Exception as e:
                # Trigger failure event
                if 'failure' in on:
                    manager.trigger('failure', func.__name__, params, error=str(e))

                raise

        wrapper.__uf_webhook_enabled__ = True
        wrapper.__uf_webhook_events__ = on
        wrapper.__uf_webhook_manager__ = manager

        return wrapper

    return decorator


# Global webhook manager
_global_webhook_manager: Optional[WebhookManager] = None


def set_global_webhook_manager(manager: WebhookManager) -> None:
    """Set the global webhook manager."""
    global _global_webhook_manager
    _global_webhook_manager = manager


def get_global_webhook_manager() -> Optional[WebhookManager]:
    """Get the global webhook manager."""
    return _global_webhook_manager
