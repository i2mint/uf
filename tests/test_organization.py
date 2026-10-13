"""Groups serve partials, callable instances and ``{name: callable}`` mappings
(i2mint/uf#24)."""

import functools

import pytest

from uf.organization import FunctionGroup, FunctionOrganizer, mk_grouped_app


def add(x: int, y: int = 1) -> int:
    "Add."
    return x + y


class Doubler:
    def __call__(self, x: int) -> int:
        return 2 * x


def test_group_names_partials_and_callable_instances():
    group = FunctionGroup("G", [functools.partial(add, y=2), Doubler()])
    assert group.get_function_names() == ["add", "Doubler"]


def test_group_accepts_a_mapping_and_add_function_names_explicitly():
    group = FunctionGroup("G", {"plus": add})
    group.add_function(Doubler())
    group.add_function(add, name="plus2")
    assert list(group.named_funcs) == ["plus", "Doubler", "plus2"]
    group = FunctionGroup("G", [add]).add_function(Doubler(), name="twice")
    assert list(group.named_funcs) == ["add", "twice"]


def test_organizer_to_dict_uses_names_and_wrapped_docs():
    organizer = FunctionOrganizer()
    organizer.group("G", {"plus": functools.partial(add, y=2)})
    (group,) = organizer.to_dict()["groups"]
    assert group["functions"] == [{"name": "plus", "description": "Add."}]
    assert organizer.get_all_functions() == [organizer.get_groups()[0].funcs["plus"]]


def test_mk_grouped_app_serves_partials_instances_and_mappings():
    pytest.importorskip("bottle")
    groups = [
        FunctionGroup("A", [functools.partial(add, y=2), Doubler()]),
        FunctionGroup("B", {"plus": add}),
    ]
    app = mk_grouped_app(groups)
    assert [g.name for g in app.function_groups] == ["A", "B"]
    assert [
        f["name"] for g in app.organization.to_dict()["groups"] for f in g["functions"]
    ] == [
        "add",
        "Doubler",
        "plus",
    ]


def test_clashing_names_across_groups_raise_and_the_same_function_is_fine():
    pytest.importorskip("bottle")
    other = functools.partial(add, y=5)
    with pytest.raises(ValueError, match="both named 'add'"):
        mk_grouped_app([FunctionGroup("A", [add]), FunctionGroup("B", [other])])
    mk_grouped_app([FunctionGroup("A", [add]), FunctionGroup("B", [add])])
