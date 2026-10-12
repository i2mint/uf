# uf.webhooks

Webhook support for uf.

Provides functionality to trigger HTTP callbacks when functions complete,
enabling integration with external services.

### Functions

| [`get_global_webhook_manager`](#uf.webhooks.get_global_webhook_manager)()        | Get the global webhook manager.          |
|--------------------------------------------------------------------------------------|------------------------------------------|
| [`set_global_webhook_manager`](#uf.webhooks.set_global_webhook_manager)(manager) | Set the global webhook manager.          |
| [`webhook`](#uf.webhooks.webhook)([on, url, manager])         | Decorator to add webhooks to a function. |

### Classes

| [`WebhookClient`](#uf.webhooks.WebhookClient)(url[, headers, timeout, ...])   | Client for sending webhooks.                  |
|------------------------------------------------------------------------------------------------|-----------------------------------------------|
| [`WebhookEvent`](#uf.webhooks.WebhookEvent)(event_type, func_name, params)   | Represents a webhook event.                   |
| [`WebhookManager`](#uf.webhooks.WebhookManager)()                              | Manage webhooks for multiple URLs and events. |

### *class* uf.webhooks.WebhookClient(url, headers=None, timeout=10.0, retry_count=3)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Client for sending webhooks.

### Example

```pycon
>>> client = WebhookClient('https://example.com/webhook')
>>> client.send(WebhookEvent('success', 'my_func', {}, result=42))
```

#### send(event, async_send=True)

Send a webhook event.

* **Parameters:**
  * **event** ([`WebhookEvent`](#uf.webhooks.WebhookEvent)) – WebhookEvent to send
  * **async_send** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to send asynchronously
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if sent successfully (for sync sends)

### *class* uf.webhooks.WebhookEvent(event_type, func_name, params, result=None, error=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Represents a webhook event.

#### event_type

Type of event (‘success’, ‘failure’, ‘start’)

#### func_name

Name of the function

#### params

Function parameters

#### result

Function result (if success)

#### error

Error message (if failure)

#### timestamp

When event occurred

#### to_dict()

Convert to dictionary for JSON serialization.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary representation

### *class* uf.webhooks.WebhookManager

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Manage webhooks for multiple URLs and events.

### Example

```pycon
>>> manager = WebhookManager()
>>> manager.add_webhook('https://example.com/hook1', events=['success'])
>>> manager.add_webhook('https://example.com/hook2', events=['failure'])
>>> manager.trigger('success', 'my_func', {}, result=42)
```

#### add_webhook(url, events=None, headers=None, condition=None)

Add a webhook.

* **Parameters:**
  * **url** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Webhook URL
  * **events** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – List of event types to trigger on (None = all)
  * **headers** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Optional HTTP headers
  * **condition** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Optional callable to filter events
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### list_webhooks()

List all registered webhooks.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]
* **Returns:**
  List of webhook configurations

#### remove_webhook(url)

Remove a webhook by URL.

* **Parameters:**
  **url** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Webhook URL
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if removed

#### trigger(event_type, func_name, params, result=None, error=None)

Trigger webhooks for an event.

* **Parameters:**
  * **event_type** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Event type
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
  * **params** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Function parameters
  * **result** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Function result
  * **error** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Error message
* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)
* **Returns:**
  Number of webhooks triggered

### uf.webhooks.get_global_webhook_manager()

Get the global webhook manager.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`WebhookManager`](#uf.webhooks.WebhookManager)]

### uf.webhooks.set_global_webhook_manager(manager)

Set the global webhook manager.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### uf.webhooks.webhook(on=None, url=None, manager=None)

Decorator to add webhooks to a function.

* **Parameters:**
  * **on** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – List of events to trigger on (‘success’, ‘failure’, ‘start’)
  * **url** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional webhook URL
  * **manager** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`WebhookManager`](#uf.webhooks.WebhookManager)]) – Optional WebhookManager instance
* **Returns:**
  Decorator function

### Example

```pycon
>>> @webhook(on=['success', 'failure'])
... def process_order(order_id: int):
...     # Process order
...     return {'status': 'processed'}
```
