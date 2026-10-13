"""Callables whose annotations are strings naming this module's own types (used by
test_specs.py); not a test module."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Pt:
    x: int = 0


def shift(p: Pt, dx: int = 1) -> int:
    return dx


class AsyncDoubler:
    async def __call__(self, x: int = 1) -> int:
        return 2 * x


class NamedAsyncTripler:
    __name__ = 'tripler'

    async def __call__(self, x: int = 1) -> int:
        return 3 * x
