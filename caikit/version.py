try:
    # Local
    from ._version import __version__, __version_tuple__  # noqa: F401 # unused import
except ImportError:
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("caikit")
    except PackageNotFoundError:
        __version__ = "unknown"
    __version_tuple__ = tuple(__version__.split("."))
