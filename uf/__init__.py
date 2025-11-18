"""uf - UI Fast: Minimal-boilerplate web UIs for Python functions.

uf bridges functions ’ HTTP services (via qh) ’ Web UI forms (via ju.rjsf),
following the "convention over configuration" philosophy.

Basic usage:
    >>> from uf import mk_rjsf_app
    >>>
    >>> def add(x: int, y: int) -> int:
    ...     '''Add two numbers'''
    ...     return x + y
    >>>
    >>> app = mk_rjsf_app([add])
    >>> # app.run()  # Start the web server

The main entry points are:
- `mk_rjsf_app`: Create a web app from functions (functional interface)
- `UfApp`: Object-oriented wrapper with additional conveniences
- `FunctionSpecStore`: Manage function specifications (advanced usage)
"""

from uf.base import mk_rjsf_app, UfApp
from uf.specs import FunctionSpecStore

__version__ = "0.0.1"

__all__ = [
    "mk_rjsf_app",
    "UfApp",
    "FunctionSpecStore",
]
