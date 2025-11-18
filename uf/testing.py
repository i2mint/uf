"""Testing utilities for uf applications.

Provides tools for testing uf apps, including test clients, fixtures,
and assertion helpers.
"""

from typing import Callable, Any, Optional
from contextlib import contextmanager
import json


class UfTestClient:
    """Test client for uf applications.

    Provides a convenient interface for testing uf apps without
    running a web server.

    Example:
        >>> from uf import mk_rjsf_app
        >>> app = mk_rjsf_app([my_func])
        >>> client = UfTestClient(app)
        >>> response = client.call_function('my_func', {'x': 10, 'y': 20})
        >>> assert response['success']
    """

    def __init__(self, app):
        """Initialize test client.

        Args:
            app: The uf application to test
        """
        self.app = app
        self.function_specs = getattr(app, 'function_specs', None)

    def list_functions(self) -> list[str]:
        """Get list of available function names.

        Returns:
            List of function name strings
        """
        if self.function_specs:
            return list(self.function_specs.keys())
        return []

    def get_spec(self, func_name: str) -> dict:
        """Get RJSF specification for a function.

        Args:
            func_name: Name of the function

        Returns:
            Function specification dict

        Raises:
            KeyError: If function not found
        """
        if not self.function_specs:
            raise ValueError("App does not have function_specs")

        return self.function_specs[func_name]

    def call_function(
        self,
        func_name: str,
        params: dict,
        *,
        expect_success: bool = True,
    ) -> dict:
        """Call a function with the given parameters.

        Args:
            func_name: Name of the function to call
            params: Dictionary of parameters
            expect_success: Whether to expect success (raises on failure)

        Returns:
            Result dictionary with 'success' and 'result' or 'error'

        Raises:
            AssertionError: If expect_success=True and call fails
        """
        if not self.function_specs:
            raise ValueError("App does not have function_specs")

        spec = self.function_specs[func_name]
        func = spec['func']

        try:
            result = func(**params)
            response = {'success': True, 'result': result}
        except Exception as e:
            response = {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
            }

        if expect_success and not response['success']:
            raise AssertionError(
                f"Function call failed: {response['error']}"
            )

        return response

    def validate_params(self, func_name: str, params: dict) -> tuple[bool, Optional[str]]:
        """Validate parameters against function schema.

        Args:
            func_name: Name of the function
            params: Parameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            spec = self.get_spec(func_name)
            schema = spec['schema']

            # Check required parameters
            required = schema.get('required', [])
            for req_param in required:
                if req_param not in params:
                    return False, f"Missing required parameter: {req_param}"

            # Basic type checking
            properties = schema.get('properties', {})
            for param_name, value in params.items():
                if param_name not in properties:
                    return False, f"Unknown parameter: {param_name}"

                param_schema = properties[param_name]
                expected_type = param_schema.get('type')

                if expected_type:
                    if not self._check_type(value, expected_type):
                        return (
                            False,
                            f"Parameter {param_name} has wrong type. "
                            f"Expected {expected_type}, got {type(value).__name__}",
                        )

            return True, None

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _check_type(self, value: Any, json_type: str) -> bool:
        """Check if value matches JSON Schema type.

        Args:
            value: Value to check
            json_type: JSON Schema type name

        Returns:
            True if types match
        """
        type_map = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict,
            'null': type(None),
        }

        expected_types = type_map.get(json_type)
        if expected_types is None:
            return True  # Unknown type, allow it

        return isinstance(value, expected_types)


class UfAppTester:
    """Context manager for testing uf apps.

    Provides a testing context with common utilities and assertions.

    Example:
        >>> with UfAppTester(app) as tester:
        ...     result = tester.submit_form('add', {'x': 10, 'y': 20})
        ...     tester.assert_success(result)
        ...     tester.assert_result_equals(result, 30)
    """

    def __init__(self, app):
        """Initialize the app tester.

        Args:
            app: The uf application to test
        """
        self.app = app
        self.client = UfTestClient(app)
        self._results_history: list[dict] = []

    def __enter__(self) -> 'UfAppTester':
        """Enter the testing context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the testing context."""
        pass

    def submit_form(self, func_name: str, form_data: dict) -> dict:
        """Simulate form submission.

        Args:
            func_name: Name of the function
            form_data: Form data as dictionary

        Returns:
            Result dictionary
        """
        result = self.client.call_function(func_name, form_data, expect_success=False)
        self._results_history.append(result)
        return result

    def assert_success(self, result: dict, message: str = ""):
        """Assert that a result indicates success.

        Args:
            result: Result dictionary
            message: Optional custom error message

        Raises:
            AssertionError: If result is not successful
        """
        msg = message or f"Expected success but got error: {result.get('error')}"
        assert result.get('success'), msg

    def assert_failure(self, result: dict, message: str = ""):
        """Assert that a result indicates failure.

        Args:
            result: Result dictionary
            message: Optional custom error message

        Raises:
            AssertionError: If result is successful
        """
        msg = message or "Expected failure but got success"
        assert not result.get('success'), msg

    def assert_result_equals(self, result: dict, expected: Any):
        """Assert that result value equals expected.

        Args:
            result: Result dictionary
            expected: Expected value

        Raises:
            AssertionError: If values don't match
        """
        self.assert_success(result)
        actual = result.get('result')
        assert actual == expected, f"Expected {expected}, got {actual}"

    def assert_error_type(self, result: dict, error_type: str):
        """Assert that error type matches.

        Args:
            result: Result dictionary
            error_type: Expected error type name

        Raises:
            AssertionError: If error types don't match
        """
        self.assert_failure(result)
        actual_type = result.get('error_type')
        assert actual_type == error_type, f"Expected {error_type}, got {actual_type}"

    def get_history(self) -> list[dict]:
        """Get history of all results.

        Returns:
            List of result dictionaries
        """
        return self._results_history.copy()


def test_ui_function(
    func: Callable,
    test_inputs: dict,
    *,
    expected_output: Any = None,
    expected_exception: Optional[type] = None,
) -> bool:
    """Test a function with form-like input.

    Args:
        func: Function to test
        test_inputs: Dictionary of test parameters
        expected_output: Expected return value (if any)
        expected_exception: Expected exception type (if any)

    Returns:
        True if test passes

    Raises:
        AssertionError: If test fails

    Example:
        >>> def add(x: int, y: int) -> int:
        ...     return x + y
        >>> test_ui_function(add, {'x': 10, 'y': 20}, expected_output=30)
        True
    """
    if expected_exception:
        try:
            func(**test_inputs)
            raise AssertionError(f"Expected {expected_exception.__name__} but no exception was raised")
        except expected_exception:
            return True
        except Exception as e:
            raise AssertionError(
                f"Expected {expected_exception.__name__} but got {type(e).__name__}: {e}"
            )
    else:
        result = func(**test_inputs)

        if expected_output is not None:
            assert result == expected_output, f"Expected {expected_output}, got {result}"

        return True


@contextmanager
def mock_function_response(app, func_name: str, mock_result: Any):
    """Context manager to mock a function's response.

    Args:
        app: The uf application
        func_name: Name of function to mock
        mock_result: Value to return

    Yields:
        None

    Example:
        >>> with mock_function_response(app, 'get_user', {'name': 'Test'}):
        ...     # Calls to get_user will return {'name': 'Test'}
        ...     result = client.call_function('get_user', {'id': 1})
    """
    if not hasattr(app, 'function_specs'):
        raise ValueError("App does not have function_specs")

    spec = app.function_specs[func_name]
    original_func = spec['func']

    # Create mock function
    def mock_func(**kwargs):
        return mock_result

    # Replace function
    spec['func'] = mock_func

    try:
        yield
    finally:
        # Restore original function
        spec['func'] = original_func


class FormDataBuilder:
    """Builder for constructing form data for tests.

    Provides a fluent interface for building test form data.

    Example:
        >>> form_data = (
        ...     FormDataBuilder()
        ...     .field('name', 'John Doe')
        ...     .field('email', 'john@example.com')
        ...     .field('age', 30)
        ...     .build()
        ... )
    """

    def __init__(self):
        """Initialize the builder."""
        self._data: dict = {}

    def field(self, name: str, value: Any) -> 'FormDataBuilder':
        """Add a field to the form data.

        Args:
            name: Field name
            value: Field value

        Returns:
            Self for method chaining
        """
        self._data[name] = value
        return self

    def fields(self, **kwargs) -> 'FormDataBuilder':
        """Add multiple fields at once.

        Args:
            **kwargs: Field name-value pairs

        Returns:
            Self for method chaining
        """
        self._data.update(kwargs)
        return self

    def build(self) -> dict:
        """Build and return the form data.

        Returns:
            Dictionary of form data
        """
        return self._data.copy()


def assert_valid_rjsf_spec(spec: dict):
    """Assert that a specification is valid RJSF format.

    Args:
        spec: The specification to validate

    Raises:
        AssertionError: If spec is invalid
    """
    assert 'schema' in spec, "Spec must have 'schema' key"
    assert isinstance(spec['schema'], dict), "Schema must be a dict"

    schema = spec['schema']
    assert 'type' in schema, "Schema must have 'type'"
    assert 'properties' in schema, "Schema must have 'properties'"
    assert isinstance(schema['properties'], dict), "Properties must be a dict"


def assert_has_field(spec: dict, field_name: str):
    """Assert that a spec has a specific field.

    Args:
        spec: The specification
        field_name: Name of the field to check

    Raises:
        AssertionError: If field not found
    """
    assert_valid_rjsf_spec(spec)
    properties = spec['schema']['properties']
    assert field_name in properties, f"Field '{field_name}' not found in schema"


def assert_field_type(spec: dict, field_name: str, expected_type: str):
    """Assert that a field has the expected type.

    Args:
        spec: The specification
        field_name: Name of the field
        expected_type: Expected JSON Schema type

    Raises:
        AssertionError: If type doesn't match
    """
    assert_has_field(spec, field_name)
    field_schema = spec['schema']['properties'][field_name]
    actual_type = field_schema.get('type')
    assert actual_type == expected_type, f"Expected type '{expected_type}', got '{actual_type}'"


def assert_field_required(spec: dict, field_name: str):
    """Assert that a field is required.

    Args:
        spec: The specification
        field_name: Name of the field

    Raises:
        AssertionError: If field is not required
    """
    assert_has_field(spec, field_name)
    required = spec['schema'].get('required', [])
    assert field_name in required, f"Field '{field_name}' is not required"
