# uf.history

Call history and presets for uf.

Provides functionality to track function calls, save parameter presets,
and reuse previous calls for improved user experience.

### Functions

| [`enable_history`](#uf.history.enable_history)(func[, max_size])   | Decorator to enable call history for a function.   |
|-------------------------------------------------------------------------------------|----------------------------------------------------|
| [`get_global_history_manager`](#uf.history.get_global_history_manager)()       | Get the global history manager.                    |

### Classes

| [`CallHistory`](#uf.history.CallHistory)([max_size])                        | Manage history of function calls.         |
|-------------------------------------------------------------------------------------------------|-------------------------------------------|
| [`FunctionCall`](#uf.history.FunctionCall)(func_name, params[, result, ...]) | Record of a single function call.         |
| [`HistoryManager`](#uf.history.HistoryManager)([max_history])                  | Combined manager for history and presets. |
| [`Preset`](#uf.history.Preset)(name, func_name, params[, description]) | A saved parameter preset for a function.  |
| [`PresetManager`](#uf.history.PresetManager)()                                | Manage parameter presets for functions.   |

### *class* uf.history.CallHistory(max_size=100)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Manage history of function calls.

### Example

```pycon
>>> history = CallHistory(max_size=100)
>>> history.record('add', {'x': 10, 'y': 20}, result=30)
>>> recent = history.get_recent('add', limit=5)
```

#### clear(func_name=None)

Clear history.

* **Parameters:**
  **func_name** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Function to clear, or None for all
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *classmethod* from_dict(data, max_size=100)

Create from dictionary.

* **Parameters:**
  * **data** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Dictionary with history data
  * **max_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum size per function
* **Return type:**
  [`CallHistory`](#uf.history.CallHistory)
* **Returns:**
  CallHistory instance

#### get_recent(func_name, limit=10)

Get recent calls for a function.

* **Parameters:**
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
  * **limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number to return
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`FunctionCall`](#uf.history.FunctionCall)]
* **Returns:**
  List of recent FunctionCall objects (newest first)

#### get_successful_calls(func_name, limit=10)

Get recent successful calls.

* **Parameters:**
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
  * **limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number to return
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`FunctionCall`](#uf.history.FunctionCall)]
* **Returns:**
  List of successful FunctionCall objects

#### record(func_name, params, result=None, success=True, error=None)

Record a function call.

* **Parameters:**
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the function
  * **params** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Parameters used
  * **result** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Result returned
  * **success** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether call succeeded
  * **error** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Error message if failed
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### to_dict()

Convert history to dictionary.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary representation

### *class* uf.history.FunctionCall(func_name, params, result=None, success=True, error=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Record of a single function call.

#### func_name

Name of the function

#### params

Parameters used

#### result

Result returned (if captured)

#### timestamp

When the call was made

#### success

Whether the call succeeded

#### error

Error message if failed

#### *classmethod* from_dict(data)

Create from dictionary.

* **Parameters:**
  **data** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Dictionary with call data
* **Return type:**
  [`FunctionCall`](#uf.history.FunctionCall)
* **Returns:**
  FunctionCall instance

#### to_dict()

Convert to dictionary.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary representation

### *class* uf.history.HistoryManager(max_history=100)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Combined manager for history and presets.

Provides a unified interface for tracking calls and managing presets.

### Example

```pycon
>>> manager = HistoryManager()
>>> manager.record_call('add', {'x': 10, 'y': 20}, result=30)
>>> _ = manager.save_preset('quick', 'add', {'x': 10, 'y': 20})
>>> recent = manager.get_recent_calls('add')
>>> presets = manager.get_presets('add')
```

#### get_preset(preset_name, func_name)

Get a preset.

* **Parameters:**
  * **preset_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of preset
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Preset`](#uf.history.Preset)]
* **Returns:**
  Preset object or None

#### get_presets(func_name)

Get all presets for a function.

* **Parameters:**
  **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Preset`](#uf.history.Preset)]
* **Returns:**
  List of Preset objects

#### get_recent_calls(func_name, limit=10)

Get recent calls.

* **Parameters:**
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
  * **limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number to return
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`FunctionCall`](#uf.history.FunctionCall)]
* **Returns:**
  List of recent FunctionCall objects

#### *classmethod* load_from_file(filepath, max_history=100)

Load from file.

* **Parameters:**
  * **filepath** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to load from
  * **max_history** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum history size
* **Return type:**
  [`HistoryManager`](#uf.history.HistoryManager)
* **Returns:**
  HistoryManager instance

#### record_call(func_name, params, result=None, success=True, error=None)

Record a function call.

* **Parameters:**
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
  * **params** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Parameters used
  * **result** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Result returned
  * **success** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether call succeeded
  * **error** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Error message if failed
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### save_preset(preset_name, func_name, params, description='')

Save a preset.

* **Parameters:**
  * **preset_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name for preset
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
  * **params** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Parameter values
  * **description** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Optional description
* **Return type:**
  [`Preset`](#uf.history.Preset)
* **Returns:**
  Created Preset object

#### save_to_file(filepath)

Save history and presets to file.

* **Parameters:**
  **filepath** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to save to
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### *class* uf.history.Preset(name, func_name, params, description='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A saved parameter preset for a function.

#### name

Preset name

#### func_name

Function this preset is for

#### params

Parameter values

#### description

Optional description

#### created_at

When preset was created

#### *classmethod* from_dict(data)

Create from dictionary.

* **Parameters:**
  **data** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Dictionary with preset data
* **Return type:**
  [`Preset`](#uf.history.Preset)
* **Returns:**
  Preset instance

#### to_dict()

Convert to dictionary.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary representation

### *class* uf.history.PresetManager

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Manage parameter presets for functions.

### Example

```pycon
>>> presets = PresetManager()
>>> _ = presets.save('quick_add', 'add', {'x': 10, 'y': 20}, 'Quick test')
>>> preset = presets.get('quick_add', 'add')
>>> def add(x, y): return x + y
>>> add(**preset.params)
30
```

#### delete(preset_name, func_name)

Delete a preset.

* **Parameters:**
  * **preset_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of preset
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if deleted, False if not found

#### *classmethod* from_dict(data)

Create from dictionary.

* **Parameters:**
  **data** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Dictionary with preset data
* **Return type:**
  [`PresetManager`](#uf.history.PresetManager)
* **Returns:**
  PresetManager instance

#### get(preset_name, func_name)

Get a preset.

* **Parameters:**
  * **preset_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of preset
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Preset`](#uf.history.Preset)]
* **Returns:**
  Preset object or None

#### list_presets(func_name)

List all presets for a function.

* **Parameters:**
  **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Preset`](#uf.history.Preset)]
* **Returns:**
  List of Preset objects

#### save(preset_name, func_name, params, description='')

Save a parameter preset.

* **Parameters:**
  * **preset_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name for this preset
  * **func_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function this preset is for
  * **params** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Parameter values
  * **description** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Optional description
* **Return type:**
  [`Preset`](#uf.history.Preset)
* **Returns:**
  Created Preset object

#### to_dict()

Convert to dictionary.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary representation

### uf.history.enable_history(func, max_size=100)

Decorator to enable call history for a function.

* **Parameters:**
  **max_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum history size
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)
* **Returns:**
  Decorator function

### Example

```pycon
>>> @enable_history
... def my_function(x: int):
...     return x * 2
```

### uf.history.get_global_history_manager()

Get the global history manager.

* **Return type:**
  [`HistoryManager`](#uf.history.HistoryManager)
* **Returns:**
  Global HistoryManager instance
