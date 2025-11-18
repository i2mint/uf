"""Tests for uf.decorators module."""

import pytest
from uf.decorators import (
    ui_config,
    group,
    hidden,
    field_config,
    get_ui_config,
    get_group,
    get_field_configs,
    is_hidden,
    with_example,
    get_example,
)
from uf.rjsf_config import RjsfFieldConfig


def test_ui_config_decorator():
    """Test ui_config decorator."""

    @ui_config(title="Test Function", group="TestGroup")
    def test_func():
        pass

    config = get_ui_config(test_func)
    assert config is not None
    assert config['title'] == "Test Function"
    assert config['group'] == "TestGroup"


def test_group_decorator():
    """Test group decorator."""

    @group("Admin")
    def admin_func():
        pass

    func_group = get_group(admin_func)
    assert func_group == "Admin"


def test_hidden_decorator():
    """Test hidden decorator."""

    @hidden
    def secret_func():
        pass

    assert is_hidden(secret_func)


def test_field_config_decorator():
    """Test field_config decorator."""
    email_config = RjsfFieldConfig(format='email')
    bio_config = RjsfFieldConfig(widget='textarea')

    @field_config(email=email_config, bio=bio_config)
    def create_profile(email: str, bio: str):
        pass

    configs = get_field_configs(create_profile)
    assert 'email' in configs
    assert 'bio' in configs
    assert configs['email'].format == 'email'
    assert configs['bio'].widget == 'textarea'


def test_with_example_decorator():
    """Test with_example decorator."""

    @with_example(x=10, y=20)
    def add(x: int, y: int):
        return x + y

    example = get_example(add)
    assert example is not None
    args, kwargs = example
    assert kwargs == {'x': 10, 'y': 20}


def test_ui_config_with_fields():
    """Test ui_config with field configurations."""
    email_config = RjsfFieldConfig(format='email')

    @ui_config(
        title="User Form",
        fields={'email': email_config}
    )
    def create_user(email: str):
        pass

    config = get_ui_config(create_user)
    assert config['fields']['email'].format == 'email'


def test_get_ui_config_none():
    """Test getting config from unconfigured function."""

    def plain_func():
        pass

    config = get_ui_config(plain_func)
    assert config is None


def test_is_hidden_false():
    """Test is_hidden on non-hidden function."""

    def visible_func():
        pass

    assert not is_hidden(visible_func)


def test_decorator_preserves_function():
    """Test that decorators preserve the original function."""

    @ui_config(title="Test")
    def test_func(x: int) -> int:
        return x * 2

    # Function should still work
    assert test_func(5) == 10
    # And should have the config
    assert get_ui_config(test_func)['title'] == "Test"
