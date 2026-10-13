"""Import optional dependencies without masking broken installs.

A bare ``except ImportError`` around ``import pandas`` also swallows the
``ImportError`` of a pandas that *is* installed but fails to import (a missing
dependency of pandas, an incompatible binary). The feature then silently turns itself
off. ``import_optional`` tolerates only the absence of the named package itself.
"""

import importlib
from types import ModuleType
from typing import Optional


def import_optional(
    module_name: str, *, absent: Optional[tuple] = None
) -> Optional[ModuleType]:
    """Return the imported module, or ``None`` if the optional package is not installed.

    Only a ``ModuleNotFoundError`` naming the package itself (the first segment of
    ``module_name``, or one of ``absent``) counts as "not installed". Every other
    failure -- a missing submodule, a missing dependency *of* the package, a broken
    import inside it -- is a broken environment and propagates.

    >>> import_optional('json').__name__
    'json'
    >>> import_optional('_uf_no_such_package_') is None
    True
    >>> import_optional('json.no_such_submodule')
    Traceback (most recent call last):
      ...
    ModuleNotFoundError: No module named 'json.no_such_submodule'
    """
    absent = absent or (module_name.split(".")[0],)
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name not in absent:
            raise
        return None
