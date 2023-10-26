# pylint: disable=unused-import
try:
    # Local
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "unknown"
    version_tuple = (0, 0, __version__)
