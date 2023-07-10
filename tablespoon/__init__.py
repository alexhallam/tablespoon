import importlib.metadata

def get_package_version():
    try:
        return importlib.metadata.version("tablespoon")
    except importlib.metadata.PackageNotFoundError:
        return None

__version__ = get_package_version()

import pkgutil

from .forecasters import Mean, Naive, Snaive
