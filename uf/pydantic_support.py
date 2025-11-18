"""Pydantic integration for uf.

Provides seamless integration with Pydantic models, automatically
generating forms with validation from Pydantic models.
"""

from typing import Callable, Any, Optional, get_type_hints, get_args, get_origin
import inspect


def is_pydantic_model(obj: Any) -> bool:
    """Check if an object is a Pydantic model.

    Args:
        obj: Object to check

    Returns:
        True if object is a Pydantic BaseModel
    """
    try:
        from pydantic import BaseModel
        if inspect.isclass(obj):
            return issubclass(obj, BaseModel)
        return isinstance(obj, BaseModel)
    except ImportError:
        return False


def pydantic_model_to_json_schema(model_class) -> dict:
    """Convert a Pydantic model to JSON Schema.

    Args:
        model_class: Pydantic model class

    Returns:
        JSON Schema dictionary

    Example:
        >>> from pydantic import BaseModel
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>> schema = pydantic_model_to_json_schema(User)
    """
    if not is_pydantic_model(model_class):
        raise ValueError(f"{model_class} is not a Pydantic model")

    # Pydantic v2 compatibility
    try:
        # Pydantic v2
        return model_class.model_json_schema()
    except AttributeError:
        # Pydantic v1
        return model_class.schema()


def function_uses_pydantic(func: Callable) -> bool:
    """Check if a function uses Pydantic models in its signature.

    Args:
        func: Function to check

    Returns:
        True if any parameter is a Pydantic model
    """
    try:
        type_hints = get_type_hints(func)
        return any(is_pydantic_model(hint) for hint in type_hints.values())
    except Exception:
        return False


def extract_pydantic_params(func: Callable) -> dict[str, Any]:
    """Extract Pydantic model parameters from function signature.

    Args:
        func: Function to analyze

    Returns:
        Dictionary mapping parameter names to Pydantic model classes
    """
    pydantic_params = {}

    try:
        type_hints = get_type_hints(func)
        for param_name, param_type in type_hints.items():
            if is_pydantic_model(param_type):
                pydantic_params[param_name] = param_type
    except Exception:
        pass

    return pydantic_params


def create_pydantic_spec(func: Callable) -> Optional[dict]:
    """Create RJSF spec from function with Pydantic parameters.

    Args:
        func: Function that uses Pydantic models

    Returns:
        RJSF specification dictionary or None

    Example:
        >>> from pydantic import BaseModel, EmailStr
        >>> class UserCreate(BaseModel):
        ...     email: EmailStr
        ...     age: int
        >>> def create_user(user: UserCreate):
        ...     pass
        >>> spec = create_pydantic_spec(create_user)
    """
    pydantic_params = extract_pydantic_params(func)

    if not pydantic_params:
        return None

    # If single Pydantic parameter, use its schema directly
    if len(pydantic_params) == 1:
        param_name, model_class = list(pydantic_params.items())[0]
        schema = pydantic_model_to_json_schema(model_class)

        # Clean up schema
        if '$defs' in schema:
            schema.pop('$defs')

        return {
            'schema': schema,
            'uiSchema': {},
            'pydantic_model': model_class,
            'param_name': param_name,
        }

    # Multiple Pydantic parameters - combine schemas
    combined_schema = {
        'type': 'object',
        'properties': {},
        'required': [],
        'title': func.__name__,
    }

    for param_name, model_class in pydantic_params.items():
        model_schema = pydantic_model_to_json_schema(model_class)
        combined_schema['properties'][param_name] = model_schema

        # Add to required if no default
        sig = inspect.signature(func)
        if sig.parameters[param_name].default == inspect.Parameter.empty:
            combined_schema['required'].append(param_name)

    return {
        'schema': combined_schema,
        'uiSchema': {},
        'pydantic_models': pydantic_params,
    }


def pydantic_to_dict(obj: Any) -> dict:
    """Convert Pydantic model instance to dictionary.

    Args:
        obj: Pydantic model instance

    Returns:
        Dictionary representation
    """
    if not is_pydantic_model(obj):
        return obj

    # Pydantic v2 compatibility
    try:
        # Pydantic v2
        return obj.model_dump()
    except AttributeError:
        # Pydantic v1
        return obj.dict()


def dict_to_pydantic(data: dict, model_class) -> Any:
    """Convert dictionary to Pydantic model instance.

    Args:
        data: Dictionary of data
        model_class: Pydantic model class

    Returns:
        Pydantic model instance

    Raises:
        ValidationError: If data doesn't match model schema
    """
    if not is_pydantic_model(model_class):
        return data

    # Pydantic v2 compatibility
    try:
        # Pydantic v2
        return model_class.model_validate(data)
    except AttributeError:
        # Pydantic v1
        return model_class.parse_obj(data)


def wrap_pydantic_function(func: Callable) -> Callable:
    """Wrap a function that uses Pydantic models.

    The wrapper converts dict inputs to Pydantic models before calling
    the function, and converts Pydantic outputs back to dicts.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function

    Example:
        >>> from pydantic import BaseModel
        >>> class User(BaseModel):
        ...     name: str
        >>> def create_user(user: User) -> User:
        ...     return user
        >>> wrapped = wrap_pydantic_function(create_user)
        >>> result = wrapped({'name': 'Alice'})  # Pass dict, not User
    """
    pydantic_params = extract_pydantic_params(func)

    if not pydantic_params:
        return func

    def wrapper(**kwargs):
        """Wrapper that handles Pydantic conversion."""
        # Convert dict inputs to Pydantic models
        converted_kwargs = {}

        for param_name, value in kwargs.items():
            if param_name in pydantic_params:
                model_class = pydantic_params[param_name]
                if isinstance(value, dict):
                    converted_kwargs[param_name] = dict_to_pydantic(
                        value, model_class
                    )
                else:
                    converted_kwargs[param_name] = value
            else:
                converted_kwargs[param_name] = value

        # Call function
        result = func(**converted_kwargs)

        # Convert Pydantic output to dict if needed
        if is_pydantic_model(result):
            return pydantic_to_dict(result)

        return result

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__uf_pydantic_wrapped__ = True
    wrapper.__uf_original_function__ = func

    return wrapper


def extract_field_validators(model_class) -> dict:
    """Extract field validators from Pydantic model.

    Args:
        model_class: Pydantic model class

    Returns:
        Dictionary mapping field names to validator info
    """
    if not is_pydantic_model(model_class):
        return {}

    validators = {}

    try:
        # Pydantic v2
        if hasattr(model_class, 'model_fields'):
            for field_name, field_info in model_class.model_fields.items():
                validator_info = {
                    'required': field_info.is_required(),
                    'default': field_info.default if field_info.default is not None else None,
                }

                # Extract constraints
                if hasattr(field_info, 'constraints'):
                    constraints = {}
                    for constraint in ['gt', 'ge', 'lt', 'le', 'min_length', 'max_length']:
                        if hasattr(field_info, constraint):
                            val = getattr(field_info, constraint)
                            if val is not None:
                                constraints[constraint] = val

                    if constraints:
                        validator_info['constraints'] = constraints

                validators[field_name] = validator_info
    except Exception:
        # Pydantic v1 or other issues
        pass

    return validators


def pydantic_error_to_user_friendly(error) -> dict:
    """Convert Pydantic ValidationError to user-friendly format.

    Args:
        error: Pydantic ValidationError

    Returns:
        Dictionary with field-level errors
    """
    try:
        from pydantic import ValidationError
    except ImportError:
        return {'error': str(error)}

    if not isinstance(error, ValidationError):
        return {'error': str(error)}

    field_errors = {}

    for err in error.errors():
        field = '.'.join(str(loc) for loc in err['loc'])
        message = err['msg']
        field_errors[field] = message

    return {'field_errors': field_errors}


class PydanticRegistry:
    """Registry for Pydantic models used in uf.

    Tracks models and provides utilities for working with them.
    """

    def __init__(self):
        """Initialize the registry."""
        self._models: dict[str, Any] = {}

    def register(self, name: str, model_class: Any) -> None:
        """Register a Pydantic model.

        Args:
            name: Name to register under
            model_class: Pydantic model class
        """
        if not is_pydantic_model(model_class):
            raise ValueError(f"{model_class} is not a Pydantic model")

        self._models[name] = model_class

    def get(self, name: str) -> Optional[Any]:
        """Get a registered model by name.

        Args:
            name: Model name

        Returns:
            Pydantic model class or None
        """
        return self._models.get(name)

    def list_models(self) -> list[str]:
        """List all registered model names.

        Returns:
            List of model name strings
        """
        return list(self._models.keys())


# Global registry
_global_pydantic_registry = PydanticRegistry()


def get_pydantic_registry() -> PydanticRegistry:
    """Get the global Pydantic registry.

    Returns:
        Global PydanticRegistry instance
    """
    return _global_pydantic_registry
