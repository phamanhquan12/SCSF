"""Version and environment pinning helpers.

Everything that must be recorded in a run manifest reasonably lives here so
the manifest writer does not depend on importing torch at module import time.
"""

__version__ = "0.1.0"

import importlib.metadata
import sys


def python_version():
    return sys.version.split()[0]


def package_versions(torch=None, torchvision=None, timm=None, numpy=None):
    """Return a dict of package versions, resolving lazily.

    Pure-python callers (e.g. aggregate) may pass already-imported modules.
    """
    out = {
        "python": python_version(),
        "scsf": __version__,
    }
    for name, mod in (
        ("torch", torch),
        ("torchvision", torchvision),
        ("timm", timm),
        ("numpy", numpy),
    ):
        if mod is None:
            try:
                mod = importlib.import_module(name)
            except Exception:
                mod = None
        if mod is not None:
            out[name] = getattr(mod, "__version__", "unknown")
    return out