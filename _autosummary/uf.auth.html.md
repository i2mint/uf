# uf.auth

Authentication and authorization for uf.

Provides authentication backends, session management, and role-based
access control for uf applications.

### Functions

| [`get_global_api_key_manager`](#uf.auth.get_global_api_key_manager)()                | Get the global API key manager.                     |
|----------------------------------------------------------------------------------------------|-----------------------------------------------------|
| [`get_global_auth_backend`](#uf.auth.get_global_auth_backend)()                   | Get the global authentication backend.              |
| [`get_global_session_manager`](#uf.auth.get_global_session_manager)()                | Get the global session manager.                     |
| [`require_auth`](#uf.auth.require_auth)(backend[, roles, permissions]) | Decorator to require authentication for a function. |
| [`set_global_api_key_manager`](#uf.auth.set_global_api_key_manager)(manager)         | Set the global API key manager.                     |
| [`set_global_auth_backend`](#uf.auth.set_global_auth_backend)(backend)            | Set the global authentication backend.              |
| [`set_global_session_manager`](#uf.auth.set_global_session_manager)(manager)         | Set the global session manager.                     |

### Classes

| [`ApiKey`](#uf.auth.ApiKey)(key, name[, permissions, expires_at])   | API key for programmatic access.                  |
|-------------------------------------------------------------------------------------------------|---------------------------------------------------|
| [`ApiKeyManager`](#uf.auth.ApiKeyManager)([key_prefix])                    | Manage API keys for programmatic access.          |
| [`AuthBackend`](#uf.auth.AuthBackend)()                                  | Base authentication backend.                      |
| [`DictAuthBackend`](#uf.auth.DictAuthBackend)()                              | Simple in-memory dictionary-based authentication. |
| [`PasswordHasher`](#uf.auth.PasswordHasher)()                               | Password hashing utilities.                       |
| [`SessionManager`](#uf.auth.SessionManager)(secret_key[, session_timeout])  | Manage user sessions.                             |
| [`User`](#uf.auth.User)(username, password_hash[, roles, ...])    | User account information.                         |

### *class* uf.auth.ApiKey(key, name, permissions=None, expires_at=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

API key for programmatic access.

#### key

The API key string

#### name

Descriptive name

#### permissions

List of allowed permissions

#### created_at

When key was created

#### expires_at

Optional expiration

#### is_active

Whether key is active

#### has_permission(permission)

Check if key has permission.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### is_expired()

Check if key is expired.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### *class* uf.auth.ApiKeyManager(key_prefix='sk_')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Manage API keys for programmatic access.

### Example

```pycon
>>> api_keys = ApiKeyManager()
>>> key = api_keys.create_key('mobile_app', permissions=['read'])
>>> print(f"Your API key: {key.key}")
>>> # Later, validate
>>> if api_keys.validate_key(key.key, 'read'):
...     # Allow access
```

#### create_key(name, permissions=None, expires_in_days=None)

Create a new API key.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Descriptive name for the key
  * **permissions** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – List of allowed permissions
  * **expires_in_days** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Optional expiration in days
* **Return type:**
  [`ApiKey`](#uf.auth.ApiKey)
* **Returns:**
  Created ApiKey object

#### list_keys()

List all API keys.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`ApiKey`](#uf.auth.ApiKey)]
* **Returns:**
  List of ApiKey objects

#### revoke_key(key)

Revoke an API key.

* **Parameters:**
  **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – API key string
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if revoked

#### validate_key(key, permission=None)

Validate an API key.

* **Parameters:**
  * **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – API key string
  * **permission** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional permission to check
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if valid

### *class* uf.auth.AuthBackend

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Base authentication backend.

Subclass this to create custom authentication backends.

#### authenticate(username, password)

Authenticate a user.

* **Parameters:**
  * **username** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Username
  * **password** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Password
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`User`](#uf.auth.User)]
* **Returns:**
  User object if authenticated, None otherwise

#### create_user(username, password, roles=None, permissions=None, \*\*metadata)

Create a new user.

* **Parameters:**
  * **username** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Username
  * **password** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Plain text password
  * **roles** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – List of roles
  * **permissions** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – List of permissions
  * **\*\*metadata** – Additional user metadata
* **Return type:**
  [`User`](#uf.auth.User)
* **Returns:**
  Created User object

#### delete_user(username)

Delete a user.

* **Parameters:**
  **username** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Username
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if deleted successfully

#### get_user(username)

Get a user by username.

* **Parameters:**
  **username** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Username
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`User`](#uf.auth.User)]
* **Returns:**
  User object or None

#### update_user(username, \*\*updates)

Update user information.

* **Parameters:**
  * **username** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Username
  * **\*\*updates** – Fields to update
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if updated successfully

### *class* uf.auth.DictAuthBackend

Bases: [`AuthBackend`](#uf.auth.AuthBackend)

Simple in-memory dictionary-based authentication.

Suitable for development and simple applications.

### Example

```pycon
>>> backend = DictAuthBackend()
>>> _ = backend.create_user('admin', 'secret', roles=['admin'])
>>> user = backend.authenticate('admin', 'secret')
```

#### authenticate(username, password)

Authenticate a user.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`User`](#uf.auth.User)]

#### create_user(username, password, roles=None, permissions=None, \*\*metadata)

Create a new user.

* **Return type:**
  [`User`](#uf.auth.User)

#### delete_user(username)

Delete a user.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### *classmethod* from_dict(users_data)

Create backend from dictionary.

* **Parameters:**
  **users_data** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Dictionary mapping usernames to user info
* **Return type:**
  [`DictAuthBackend`](#uf.auth.DictAuthBackend)

### Example

```pycon
>>> backend = DictAuthBackend.from_dict({
...     'admin': {'password': 'secret', 'roles': ['admin']},
...     'user': {'password': 'pass', 'roles': ['user']},
... })
```

#### get_user(username)

Get a user by username.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`User`](#uf.auth.User)]

#### update_user(username, \*\*updates)

Update user information.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### *class* uf.auth.PasswordHasher

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Password hashing utilities.

Uses PBKDF2-HMAC-SHA256 for secure password hashing.

#### *static* hash_password(password, salt=None)

Hash a password.

* **Parameters:**
  * **password** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Plain text password
  * **salt** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional salt (generated if not provided)
* **Returns:**
  salt$hash
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### *static* verify_password(password, password_hash)

Verify a password against its hash.

* **Parameters:**
  * **password** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Plain text password
  * **password_hash** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Hash string from hash_password()
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if password matches

### *class* uf.auth.SessionManager(secret_key, session_timeout=3600)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Manage user sessions.

### Example

```pycon
>>> sessions = SessionManager(secret_key='my-secret')
>>> session_id = sessions.create_session('admin')
>>> user = sessions.get_session(session_id)
```

#### cleanup_expired()

Remove expired sessions.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)
* **Returns:**
  Number of sessions removed

#### create_session(username, data=None)

Create a new session.

* **Parameters:**
  * **username** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Username for session
  * **data** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Optional session data
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Session ID

#### delete_session(session_id)

Delete a session.

* **Parameters:**
  **session_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Session ID
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if deleted

#### get_session(session_id)

Get session data.

* **Parameters:**
  **session_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Session ID
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]
* **Returns:**
  Session data or None if expired/invalid

### *class* uf.auth.User(username, password_hash, roles=list(), permissions=list(), metadata=dict(), created_at=datetime.now(), is_active=True)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

User account information.

#### username

Unique username

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### password_hash

Hashed password

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### roles

List of role names

* **Type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### permissions

List of permission strings

* **Type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### metadata

Additional user metadata

* **Type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

#### created_at

When user was created

* **Type:**
  [`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime)

#### is_active

Whether account is active

* **Type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### has_all_roles(roles)

Check if user has all of the given roles.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### has_any_role(roles)

Check if user has any of the given roles.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### has_permission(permission)

Check if user has a permission.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### has_role(role)

Check if user has a role.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### uf.auth.get_global_api_key_manager()

Get the global API key manager.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`ApiKeyManager`](#uf.auth.ApiKeyManager)]

### uf.auth.get_global_auth_backend()

Get the global authentication backend.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`AuthBackend`](#uf.auth.AuthBackend)]

### uf.auth.get_global_session_manager()

Get the global session manager.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`SessionManager`](#uf.auth.SessionManager)]

### uf.auth.require_auth(backend, roles=None, permissions=None)

Decorator to require authentication for a function.

* **Parameters:**
  * **backend** ([`AuthBackend`](#uf.auth.AuthBackend)) – Authentication backend
  * **roles** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Required roles (any)
  * **permissions** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Required permissions (all)
* **Returns:**
  Decorator function

### Example

```pycon
>>> backend = DictAuthBackend.from_dict({
...     'admin': {'password': 'secret', 'roles': ['admin']}
... })
>>> @require_auth(backend, roles=['admin'])
... def delete_all():
...     pass
```

### uf.auth.set_global_api_key_manager(manager)

Set the global API key manager.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### uf.auth.set_global_auth_backend(backend)

Set the global authentication backend.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### uf.auth.set_global_session_manager(manager)

Set the global session manager.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)
