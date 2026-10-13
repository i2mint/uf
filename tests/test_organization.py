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


def test_the_same_bound_method_in_two_groups_is_not_a_clash():
    pytest.importorskip("bottle")

    class C:
        def meth(self, x: int) -> int:
            return x

    c = C()
    mk_grouped_app([FunctionGroup("A", [c.meth]), FunctionGroup("B", [c.meth])])


def test_generator_funcs_are_materialized():
    group = FunctionGroup("G", (f for f in [add, Doubler()]))
    assert (
        group.get_function_names() == group.get_function_names() == ["add", "Doubler"]
    )


def test_add_function_refuses_a_different_callable_with_a_taken_name():
    other = functools.partial(add, y=5)
    for funcs in ([add], {"add": add}):
        with pytest.raises(ValueError, match="already has a different callable"):
            FunctionGroup("G", funcs).add_function(other)
    FunctionGroup("G", [add]).add_function(add)
    FunctionGroup("G", [add]).add_function(other, name="add5")


# --- auto_group_by_module / auto_group_by_tag unwrap partials (i2mint/uf#26) ---

from uf.organization import auto_group_by_module, auto_group_by_tag


def _group_names(organizer):
    return {g.name: g.get_function_names() for g in organizer.get_groups()}


def test_auto_group_by_module_puts_a_partial_with_the_function_it_wraps():
    groups = _group_names(auto_group_by_module([len, functools.partial(add, y=2)]))
    assert groups == {"Builtins": ["len"], "Test Organization": ["add"]}  # no 'Functools' group


def test_auto_group_by_tag_reads_the_tag_of_the_wrapped_function():
    def tagged(x: int) -> int:
        return x

    tagged.__uf_group__ = "Admin"
    assert _group_names(auto_group_by_tag([functools.partial(tagged, 1)])) == {"Admin": ["tagged"]}


def test_a_tag_on_the_partial_itself_wins():
    def tagged(x: int) -> int:
        return x

    tagged.__uf_group__ = "Admin"
    p = functools.partial(tagged, 1)
    p.__uf_group__ = "Mine"
    assert _group_names(auto_group_by_tag([p])) == {"Mine": ["tagged"]}


def test_an_untagged_partial_still_goes_to_other():
    assert _group_names(auto_group_by_tag([functools.partial(add)])) == {"Other": ["add"]}


def test_the_outermost_partial_with_a_tag_wins_even_a_falsy_one():
    def tagged(x: int, y: int, z: int) -> int:
        return x

    tagged.__uf_group__ = "Admin"
    mid = functools.partial(tagged, 1)
    mid.__uf_group__ = "Mid"
    outer = functools.partial(mid, 2)
    assert _group_names(auto_group_by_tag([outer])) == {"Mid": ["tagged"]}
    mid.__uf_group__ = ""
    assert _group_names(auto_group_by_tag([outer])) == {"": ["tagged"]}
