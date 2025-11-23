"""Pytest configuration and lightweight asyncio support.

The execution environment for these kata-style challenges blocks network access,
which means installing ``pytest-asyncio`` (the usual way to run async tests) is
not possible. Pytest fails fast because the ``--asyncio-mode`` option in
``pytest.ini`` is unknown, and async test coroutines would never be awaited.

To keep the test suite runnable without external dependencies, we register a
compatibility shim that:

* Accepts the ``--asyncio-mode`` CLI flag so pytest startup does not abort.
* Detects coroutine test functions and executes them on a fresh event loop.

This mirrors the minimal behaviour we need from ``pytest-asyncio`` for the
suite in ``tests/api``.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register a no-op ``--asyncio-mode`` option for compatibility."""

    parser.addoption(
        "--asyncio-mode",
        action="store",
        default="auto",
        help="Compat shim for async tests without pytest-asyncio",
    )


def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Execute coroutine test functions on an event loop.

    When a collected test function is a coroutine, run it to completion on a
    dedicated event loop. Returning ``True`` tells pytest the call was handled,
    preventing the default (which would error on an un-awaited coroutine).
    """

    test_obj = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_obj):
        return None

    bound_args = {
        name: value
        for name, value in pyfuncitem.funcargs.items()
        if name in inspect.signature(test_obj).parameters
    }

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(test_obj(**bound_args))
    finally:
        loop.close()
        asyncio.set_event_loop(None)

    return True