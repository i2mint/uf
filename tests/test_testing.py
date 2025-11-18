"""Tests for uf.testing module."""

import pytest
from uf.testing import (
    test_ui_function,
    FormDataBuilder,
    assert_valid_rjsf_spec,
    assert_has_field,
    assert_field_type,
    assert_field_required,
)


def test_test_ui_function_success():
    """Test test_ui_function with successful call."""

    def add(x: int, y: int) -> int:
        return x + y

    result = test_ui_function(add, {'x': 10, 'y': 20}, expected_output=30)
    assert result is True


def test_test_ui_function_wrong_output():
    """Test test_ui_function with wrong expected output."""

    def add(x: int, y: int) -> int:
        return x + y

    with pytest.raises(AssertionError):
        test_ui_function(add, {'x': 10, 'y': 20}, expected_output=999)


def test_test_ui_function_exception():
    """Test test_ui_function expecting an exception."""

    def divide(x: int, y: int) -> float:
        return x / y

    result = test_ui_function(
        divide,
        {'x': 10, 'y': 0},
        expected_exception=ZeroDivisionError
    )
    assert result is True


def test_form_data_builder():
    """Test FormDataBuilder."""
    data = (
        FormDataBuilder()
        .field('name', 'John')
        .field('age', 30)
        .fields(email='john@example.com', city='NYC')
        .build()
    )

    assert data['name'] == 'John'
    assert data['age'] == 30
    assert data['email'] == 'john@example.com'
    assert data['city'] == 'NYC'


def test_assert_valid_rjsf_spec():
    """Test assert_valid_rjsf_spec with valid spec."""
    spec = {
        'schema': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'}
            }
        },
        'uiSchema': {}
    }

    assert_valid_rjsf_spec(spec)  # Should not raise


def test_assert_valid_rjsf_spec_invalid():
    """Test assert_valid_rjsf_spec with invalid spec."""
    spec = {'schema': {}}  # Missing required fields

    with pytest.raises(AssertionError):
        assert_valid_rjsf_spec(spec)


def test_assert_has_field():
    """Test assert_has_field."""
    spec = {
        'schema': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'age': {'type': 'integer'}
            }
        }
    }

    assert_has_field(spec, 'name')  # Should not raise
    assert_has_field(spec, 'age')  # Should not raise


def test_assert_has_field_missing():
    """Test assert_has_field with missing field."""
    spec = {
        'schema': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'}
            }
        }
    }

    with pytest.raises(AssertionError):
        assert_has_field(spec, 'nonexistent')


def test_assert_field_type():
    """Test assert_field_type."""
    spec = {
        'schema': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'age': {'type': 'integer'}
            }
        }
    }

    assert_field_type(spec, 'name', 'string')
    assert_field_type(spec, 'age', 'integer')


def test_assert_field_type_wrong():
    """Test assert_field_type with wrong type."""
    spec = {
        'schema': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'}
            }
        }
    }

    with pytest.raises(AssertionError):
        assert_field_type(spec, 'name', 'integer')


def test_assert_field_required():
    """Test assert_field_required."""
    spec = {
        'schema': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'age': {'type': 'integer'}
            },
            'required': ['name']
        }
    }

    assert_field_required(spec, 'name')  # Should not raise


def test_assert_field_required_not_required():
    """Test assert_field_required on optional field."""
    spec = {
        'schema': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'}
            },
            'required': []
        }
    }

    with pytest.raises(AssertionError):
        assert_field_required(spec, 'name')
