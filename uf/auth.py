"""Authentication and authorization for uf.

Provides authentication backends, session management, and role-based
access control for uf applications.
"""

from typing import Callable, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import secrets
import hmac
from functools import wraps


@dataclass
class User:
    """User account information.

    Attributes:
        username: Unique username
        password_hash: Hashed password
        roles: List of role names
        permissions: List of permission strings
        metadata: Additional user metadata
        created_at: When user was created
        is_active: Whether account is active
    """

    username: str
    password_hash: str
    roles: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True

    def has_role(self, role: str) -> bool:
        """Check if user has a role."""
        return role in self.roles

    def has_permission(self, permission: str) -> bool:
        """Check if user has a permission."""
        return permission in self.permissions

    def has_any_role(self, roles: list[str]) -> bool:
        """Check if user has any of the given roles."""
        return any(role in self.roles for role in roles)

    def has_all_roles(self, roles: list[str]) -> bool:
        """Check if user has all of the given roles."""
        return all(role in self.roles for role in roles)


class PasswordHasher:
    """Password hashing utilities.

    Uses PBKDF2-HMAC-SHA256 for secure password hashing.
    """

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> str:
        """Hash a password.

        Args:
            password: Plain text password
            salt: Optional salt (generated if not provided)

        Returns:
            Hash string in format: salt$hash
        """
        if salt is None:
            salt = secrets.token_hex(16)

        pwd_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )

        return f"{salt}${pwd_hash.hex()}"

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify a password against its hash.

        Args:
            password: Plain text password
            password_hash: Hash string from hash_password()

        Returns:
            True if password matches
        """
        try:
            salt, stored_hash = password_hash.split('$')
            new_hash = PasswordHasher.hash_password(password, salt)
            return hmac.compare_digest(new_hash, password_hash)
        except ValueError:
            return False


class AuthBackend:
    """Base authentication backend.

    Subclass this to create custom authentication backends.
    """

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user.

        Args:
            username: Username
            password: Password

        Returns:
            User object if authenticated, None otherwise
        """
        raise NotImplementedError

    def get_user(self, username: str) -> Optional[User]:
        """Get a user by username.

        Args:
            username: Username

        Returns:
            User object or None
        """
        raise NotImplementedError

    def create_user(
        self,
        username: str,
        password: str,
        roles: Optional[list[str]] = None,
        permissions: Optional[list[str]] = None,
        **metadata
    ) -> User:
        """Create a new user.

        Args:
            username: Username
            password: Plain text password
            roles: List of roles
            permissions: List of permissions
            **metadata: Additional user metadata

        Returns:
            Created User object
        """
        raise NotImplementedError

    def update_user(self, username: str, **updates) -> bool:
        """Update user information.

        Args:
            username: Username
            **updates: Fields to update

        Returns:
            True if updated successfully
        """
        raise NotImplementedError

    def delete_user(self, username: str) -> bool:
        """Delete a user.

        Args:
            username: Username

        Returns:
            True if deleted successfully
        """
        raise NotImplementedError


class DictAuthBackend(AuthBackend):
    """Simple in-memory dictionary-based authentication.

    Suitable for development and simple applications.

    Example:
        >>> backend = DictAuthBackend()
        >>> backend.create_user('admin', 'secret', roles=['admin'])
        >>> user = backend.authenticate('admin', 'secret')
    """

    def __init__(self):
        """Initialize the backend."""
        self._users: dict[str, User] = {}
        self._hasher = PasswordHasher()

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user."""
        user = self._users.get(username)
        if not user or not user.is_active:
            return None

        if self._hasher.verify_password(password, user.password_hash):
            return user

        return None

    def get_user(self, username: str) -> Optional[User]:
        """Get a user by username."""
        return self._users.get(username)

    def create_user(
        self,
        username: str,
        password: str,
        roles: Optional[list[str]] = None,
        permissions: Optional[list[str]] = None,
        **metadata
    ) -> User:
        """Create a new user."""
        if username in self._users:
            raise ValueError(f"User '{username}' already exists")

        password_hash = self._hasher.hash_password(password)

        user = User(
            username=username,
            password_hash=password_hash,
            roles=roles or [],
            permissions=permissions or [],
            metadata=metadata,
        )

        self._users[username] = user
        return user

    def update_user(self, username: str, **updates) -> bool:
        """Update user information."""
        user = self._users.get(username)
        if not user:
            return False

        for key, value in updates.items():
            if key == 'password':
                user.password_hash = self._hasher.hash_password(value)
            elif hasattr(user, key):
                setattr(user, key, value)

        return True

    def delete_user(self, username: str) -> bool:
        """Delete a user."""
        if username in self._users:
            del self._users[username]
            return True
        return False

    @classmethod
    def from_dict(cls, users_data: dict) -> 'DictAuthBackend':
        """Create backend from dictionary.

        Args:
            users_data: Dictionary mapping usernames to user info

        Example:
            >>> backend = DictAuthBackend.from_dict({
            ...     'admin': {'password': 'secret', 'roles': ['admin']},
            ...     'user': {'password': 'pass', 'roles': ['user']},
            ... })
        """
        backend = cls()
        hasher = PasswordHasher()

        for username, user_info in users_data.items():
            password = user_info.pop('password')
            password_hash = user_info.pop('password_hash', None)

            if password_hash is None:
                password_hash = hasher.hash_password(password)

            user = User(
                username=username,
                password_hash=password_hash,
                **user_info
            )
            backend._users[username] = user

        return backend


class SessionManager:
    """Manage user sessions.

    Example:
        >>> sessions = SessionManager(secret_key='my-secret')
        >>> session_id = sessions.create_session('admin')
        >>> user = sessions.get_session(session_id)
    """

    def __init__(self, secret_key: str, session_timeout: int = 3600):
        """Initialize session manager.

        Args:
            secret_key: Secret key for session signing
            session_timeout: Session timeout in seconds (default: 1 hour)
        """
        self.secret_key = secret_key
        self.session_timeout = session_timeout
        self._sessions: dict[str, dict] = {}

    def create_session(self, username: str, data: Optional[dict] = None) -> str:
        """Create a new session.

        Args:
            username: Username for session
            data: Optional session data

        Returns:
            Session ID
        """
        session_id = secrets.token_urlsafe(32)

        self._sessions[session_id] = {
            'username': username,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(seconds=self.session_timeout),
            'data': data or {},
        }

        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session data.

        Args:
            session_id: Session ID

        Returns:
            Session data or None if expired/invalid
        """
        session = self._sessions.get(session_id)
        if not session:
            return None

        # Check expiration
        if datetime.now() > session['expires_at']:
            del self._sessions[session_id]
            return None

        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session ID

        Returns:
            True if deleted
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def cleanup_expired(self) -> int:
        """Remove expired sessions.

        Returns:
            Number of sessions removed
        """
        now = datetime.now()
        expired = [
            sid for sid, session in self._sessions.items()
            if now > session['expires_at']
        ]

        for sid in expired:
            del self._sessions[sid]

        return len(expired)


class ApiKey:
    """API key for programmatic access.

    Attributes:
        key: The API key string
        name: Descriptive name
        permissions: List of allowed permissions
        created_at: When key was created
        expires_at: Optional expiration
        is_active: Whether key is active
    """

    def __init__(
        self,
        key: str,
        name: str,
        permissions: Optional[list[str]] = None,
        expires_at: Optional[datetime] = None,
    ):
        """Initialize API key."""
        self.key = key
        self.name = name
        self.permissions = permissions or []
        self.created_at = datetime.now()
        self.expires_at = expires_at
        self.is_active = True

    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def has_permission(self, permission: str) -> bool:
        """Check if key has permission."""
        return permission in self.permissions


class ApiKeyManager:
    """Manage API keys for programmatic access.

    Example:
        >>> api_keys = ApiKeyManager()
        >>> key = api_keys.create_key('mobile_app', permissions=['read'])
        >>> print(f"Your API key: {key.key}")
        >>> # Later, validate
        >>> if api_keys.validate_key(key.key, 'read'):
        ...     # Allow access
    """

    def __init__(self, key_prefix: str = 'sk_'):
        """Initialize API key manager.

        Args:
            key_prefix: Prefix for generated keys
        """
        self.key_prefix = key_prefix
        self._keys: dict[str, ApiKey] = {}

    def create_key(
        self,
        name: str,
        permissions: Optional[list[str]] = None,
        expires_in_days: Optional[int] = None,
    ) -> ApiKey:
        """Create a new API key.

        Args:
            name: Descriptive name for the key
            permissions: List of allowed permissions
            expires_in_days: Optional expiration in days

        Returns:
            Created ApiKey object
        """
        key_str = f"{self.key_prefix}{secrets.token_urlsafe(32)}"

        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        api_key = ApiKey(
            key=key_str,
            name=name,
            permissions=permissions,
            expires_at=expires_at,
        )

        self._keys[key_str] = api_key
        return api_key

    def validate_key(self, key: str, permission: Optional[str] = None) -> bool:
        """Validate an API key.

        Args:
            key: API key string
            permission: Optional permission to check

        Returns:
            True if valid
        """
        api_key = self._keys.get(key)
        if not api_key or not api_key.is_active:
            return False

        if api_key.is_expired():
            return False

        if permission and not api_key.has_permission(permission):
            return False

        return True

    def revoke_key(self, key: str) -> bool:
        """Revoke an API key.

        Args:
            key: API key string

        Returns:
            True if revoked
        """
        api_key = self._keys.get(key)
        if api_key:
            api_key.is_active = False
            return True
        return False

    def list_keys(self) -> list[ApiKey]:
        """List all API keys.

        Returns:
            List of ApiKey objects
        """
        return list(self._keys.values())


def require_auth(
    backend: AuthBackend,
    roles: Optional[list[str]] = None,
    permissions: Optional[list[str]] = None,
):
    """Decorator to require authentication for a function.

    Args:
        backend: Authentication backend
        roles: Required roles (any)
        permissions: Required permissions (all)

    Returns:
        Decorator function

    Example:
        >>> backend = DictAuthBackend.from_dict({
        ...     'admin': {'password': 'secret', 'roles': ['admin']}
        ... })
        >>> @require_auth(backend, roles=['admin'])
        ... def delete_all():
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This is metadata-only in current implementation
            # Actual enforcement would require middleware integration
            return func(*args, **kwargs)

        wrapper.__uf_auth_required__ = True
        wrapper.__uf_auth_roles__ = roles or []
        wrapper.__uf_auth_permissions__ = permissions or []
        wrapper.__uf_auth_backend__ = backend

        return wrapper

    return decorator


# Global instances for convenience
_global_auth_backend: Optional[AuthBackend] = None
_global_session_manager: Optional[SessionManager] = None
_global_api_key_manager: Optional[ApiKeyManager] = None


def set_global_auth_backend(backend: AuthBackend) -> None:
    """Set the global authentication backend."""
    global _global_auth_backend
    _global_auth_backend = backend


def get_global_auth_backend() -> Optional[AuthBackend]:
    """Get the global authentication backend."""
    return _global_auth_backend


def set_global_session_manager(manager: SessionManager) -> None:
    """Set the global session manager."""
    global _global_session_manager
    _global_session_manager = manager


def get_global_session_manager() -> Optional[SessionManager]:
    """Get the global session manager."""
    return _global_session_manager


def set_global_api_key_manager(manager: ApiKeyManager) -> None:
    """Set the global API key manager."""
    global _global_api_key_manager
    _global_api_key_manager = manager


def get_global_api_key_manager() -> Optional[ApiKeyManager]:
    """Get the global API key manager."""
    return _global_api_key_manager
