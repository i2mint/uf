# uf.organization

Function grouping and organization for uf.

Provides tools for organizing functions into categories, groups, and
hierarchies for better navigation in the UI.

### Functions

| [`auto_group_by_module`](#uf.organization.auto_group_by_module)(funcs)                    | Automatically group functions by their module.    |
|-------------------------------------------------------------------------------------------------|---------------------------------------------------|
| [`auto_group_by_prefix`](#uf.organization.auto_group_by_prefix)(funcs[, separator])       | Automatically group functions by name prefix.     |
| [`auto_group_by_tag`](#uf.organization.auto_group_by_tag)(funcs[, tag_attr])           | Automatically group functions by a tag attribute. |
| [`mk_grouped_app`](#uf.organization.mk_grouped_app)(groups, \*\*mk_rjsf_app_kwargs) | Create a uf app with grouped function navigation. |

### Classes

| [`FunctionGroup`](#uf.organization.FunctionGroup)(name[, funcs, description, ...])   | Group of functions with metadata.               |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------|
| [`FunctionOrganizer`](#uf.organization.FunctionOrganizer)()                              | Organize functions into groups and hierarchies. |

### *class* uf.organization.FunctionGroup(name, funcs=list(), description='', icon=None, order=0, collapsed=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Group of functions with metadata.

#### name

Name of the group

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### funcs

Functions in this group: callables (partials and callable instances
are fine; they are named as `uf.specs.name_of_func` does), or a
`{name: callable}` mapping to name them explicitly

* **Type:**
  [`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)], [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]]

#### description

Description of the group

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### icon

Optional icon identifier for the group

* **Type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### order

Display order (lower numbers first)

* **Type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

#### collapsed

Whether the group starts collapsed in UI

* **Type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### Example

```pycon
>>> math_funcs = FunctionGroup(
...     'Math',
...     [add, subtract, multiply, divide],
...     description='Mathematical operations',
...     icon='calculator'
... )
```

#### add_function(func, , name=None)

Add a function to this group.

* **Parameters:**
  * **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to add
  * **name** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Explicit name to serve it under (default: its own name).
    Giving one turns `funcs` into a `{name: callable}` mapping.
* **Return type:**
  [`FunctionGroup`](#uf.organization.FunctionGroup)
* **Returns:**
  Self for method chaining
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If a different callable already has that name in the group

#### get_function_names()

Get names of all functions in this group.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  List of function name strings

#### *property* named_funcs *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)]*

`{name: callable}` of this group, named the way the app serves them.

```pycon
>>> import functools
>>> g = FunctionGroup('G', [len, functools.partial(int, base=2)])
>>> list(g.named_funcs)
['len', 'int']
```

### *class* uf.organization.FunctionOrganizer

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Organize functions into groups and hierarchies.

This class provides a fluent interface for building function
organization structures that can be used to generate grouped
navigation in the UI.

### Example

```pycon
>>> organizer = FunctionOrganizer()
>>> organizer.group('Admin', [user_create, user_delete], icon='shield')
>>> organizer.group('Reports', [generate_report, export_csv], icon='file')
>>> groups = organizer.get_groups()
```

#### add_to_group(group_name, func)

Add a function to an existing group.

* **Parameters:**
  * **group_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the group
  * **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to add
* **Return type:**
  [`FunctionOrganizer`](#uf.organization.FunctionOrganizer)
* **Returns:**
  Self for method chaining
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If group doesn’t exist

#### add_ungrouped(func)

Add a function without a group.

* **Parameters:**
  **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to add
* **Return type:**
  [`FunctionOrganizer`](#uf.organization.FunctionOrganizer)
* **Returns:**
  Self for method chaining

#### get_all_functions()

Get all functions across all groups.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]
* **Returns:**
  List of all functions

#### get_groups()

Get all groups, sorted by order.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`FunctionGroup`](#uf.organization.FunctionGroup)]
* **Returns:**
  List of FunctionGroup objects sorted by order

#### group(name, funcs=None, , description='', icon=None, order=0, collapsed=False)

Create and add a function group.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the group
  * **funcs** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]]) – Optional functions to add to group
  * **description** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Description of the group
  * **icon** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional icon identifier
  * **order** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Display order
  * **collapsed** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to start collapsed
* **Return type:**
  [`FunctionGroup`](#uf.organization.FunctionGroup)
* **Returns:**
  The created FunctionGroup

### Example

```pycon
>>> organizer.group(
...     'Database',
...     [save_record, load_record],
...     description='Database operations',
...     icon='database'
... )
```

#### to_dict()

Convert organization to dictionary format.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary representation suitable for JSON serialization

### Example

```pycon
>>> org_dict = organizer.to_dict()
>>> # Can be used in templates or APIs
```

### uf.organization.auto_group_by_module(funcs)

Automatically group functions by their module.

* **Parameters:**
  **funcs** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Functions to organize
* **Return type:**
  [`FunctionOrganizer`](#uf.organization.FunctionOrganizer)
* **Returns:**
  FunctionOrganizer with module-based groups (a `functools.partial` goes with the
  module of the function it wraps)

### Example

```pycon
>>> from myapp import user_ops, report_ops
>>> funcs = [user_ops.create, user_ops.delete, report_ops.generate]
>>> organizer = auto_group_by_module(funcs)
```

### uf.organization.auto_group_by_prefix(funcs, separator='_')

Automatically group functions by name prefix.

Groups functions based on the part of their name before the separator.
For example, with separator=”_”:

- user_create, user_delete → “user” group
- report_generate, report_export → “report” group

* **Parameters:**
  * **funcs** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Functions to organize
  * **separator** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Separator character (default: “_”)
* **Return type:**
  [`FunctionOrganizer`](#uf.organization.FunctionOrganizer)
* **Returns:**
  FunctionOrganizer with auto-generated groups

### Example

```pycon
>>> funcs = [user_create, user_delete, report_generate, admin_reset]
>>> organizer = auto_group_by_prefix(funcs)
>>> # Creates groups: 'user', 'report', 'admin'
```

### uf.organization.auto_group_by_tag(funcs, tag_attr='_\_uf_group_\_')

Automatically group functions by a tag attribute.

Functions can be tagged with a group name using an attribute.

* **Parameters:**
  * **funcs** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Functions to organize
  * **tag_attr** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the attribute to use for grouping
* **Return type:**
  [`FunctionOrganizer`](#uf.organization.FunctionOrganizer)
* **Returns:**
  FunctionOrganizer with tag-based groups

### Example

```pycon
>>> def create_user(name: str):
...     pass
>>> create_user.__uf_group__ = 'Admin'
>>>
>>> organizer = auto_group_by_tag([create_user, other_func])
```

### uf.organization.mk_grouped_app(groups, \*\*mk_rjsf_app_kwargs)

Create a uf app with grouped function navigation.

* **Parameters:**
  * **groups** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`FunctionGroup`](#uf.organization.FunctionGroup)]) – Iterable of FunctionGroup objects
  * **\*\*mk_rjsf_app_kwargs** – Arguments passed to mk_rjsf_app
* **Returns:**
  Configured web application with grouped navigation

### Example

```pycon
>>> admin_group = FunctionGroup('Admin', [user_create, user_delete])
>>> reports_group = FunctionGroup('Reports', [generate_report])
>>> app = mk_grouped_app([admin_group, reports_group])
```
