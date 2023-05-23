"""
Handle import of TypedDict for different python versions.

We use the mypy-compatible import recommended here:
https://mypy.readthedocs.io/en/stable/runtime_troubles.html#using-new-additions-to-the-typing-module
"""

import sys

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

__all__ = ["TypedDict"]
