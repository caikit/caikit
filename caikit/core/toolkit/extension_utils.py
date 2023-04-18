# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
# pylint: disable=deprecated-module
from imp import new_module
from importlib import import_module
from types import ModuleType

# First Party
import alog

# Local
from .errors import error_handler

log = alog.use_channel("TLKIT")
error = error_handler.get(log)

EXTENSIONS_ATTR_NAME = "extensions"


def enable_extension_binding(lib_handle):
    """Idempotent initialization of extensions subpackage under a provided package. Assuming
    one doesn't exist, binds an empty "extensions" subpackage to the lib_handle. We can then
    bind individual extensions to the library.

    This should be called in the top level package initialization of the library being extended
    to enable extension module support by attribute binding.

    Args:
        lib_handle: Module
            caikit * library on which we would like to create an empty extensions subpackage,
            accessible by <lib_handle>.extensions
    """
    error.type_check("<COR64728193E>", ModuleType, lib_handle=lib_handle)
    # Ensure that we have a caikit * like module to start with, and that
    # it's initialized dynamic extension package binding.
    error.value_check(
        "<COR10841029E>",
        is_extension_like(lib_handle),
        "lib_handle must be a caikit * like Module",
    )
    if not hasattr(lib_handle, EXTENSIONS_ATTR_NAME):
        extension_pkg = new_module(EXTENSIONS_ATTR_NAME)
        setattr(lib_handle, EXTENSIONS_ATTR_NAME, extension_pkg)


def is_extension_like(lib_handle):
    """Given a handle to a module, determine if it is caikit * like. Currently, this is checked
    to see if <lib_handle>.lib_config.library_version is a string type.

    Args:
        lib_handle: Module
            caikit * library on which we would like to create an empty extensions subpackage,
            accessible by <lib_handle>.extensions

    Returns:
        bool
            True if the library handle is caikit * like.
    """
    error.type_check("<COR64748191E>", ModuleType, lib_handle=lib_handle)
    lib_version = getattr(
        getattr(lib_handle, "lib_config", None), "library_version", None
    )
    return isinstance(lib_version, str)


def bind_extensions(extension_names, lib_handle):
    """Given an iterable of extension names and a handle to a caikit library, consider each
    extension name. For each, import it & ensure it looks like a caikit library by checking to
    see if it have a lib_config attribute with a defined library version.

    For all caikit * like libraries, bind them to an extensions package on the library.

    Ex)
        Say we provide inputs:
            extension_names = ["sample_module"], a caikit extension
            lib_handle = caikit_nlp

        Register sample_module onto caikit_nlp.extensions.sample_module.

    Args:
        extension_names: list | tuple | set
            Iterable of (string) module names to be imported and bound. Objects are presumed
            to be unique & will be cast to a set for consideration. If an existing extension
            of the same name is already registered, the extension will be skipped.
        lib_handle: Module
            caikit * library on which we would like to create an empty extensions subpackage,
            accessible by <lib_handle>.extensions
    """
    error.type_check(
        "<COR64140091E>", set, list, tuple, extension_names=extension_names
    )
    error.type_check_all("<COR34341191E>", str, extension_names=extension_names)
    error.type_check("<COR64748191E>", ModuleType, lib_handle=lib_handle)
    error.value_check(
        "<COR12831731E>",
        hasattr(lib_handle, EXTENSIONS_ATTR_NAME),
        "Library {} has not enabled extension binding",
        lib_handle.__name__,
    )

    ext_subpkg = getattr(lib_handle, EXTENSIONS_ATTR_NAME)
    for ext_name in set(extension_names):
        # Skip any extension package name that's already registered
        if hasattr(ext_subpkg, ext_name):
            log.debug("Extension [{}] is already bound".format(ext_name))
            continue
        # Dynamically import each extension module; skip if we can't import
        try:
            ext_lib = import_module(ext_name)
        except ImportError:
            log.warning(
                "<COR81130101W>",
                "Module [{}] is not importable and could not be bound".format(ext_name),
            )
            continue
        # If this is a caikit * like library, bind it to the extensions property
        if is_extension_like(ext_lib):
            setattr(ext_subpkg, ext_name, ext_lib)
            log.debug("Bound extension [{}] successfully".format(ext_name))
        else:
            log.warning(
                "<COR81130101W>",
                "Skipping binding for [{}]; module is not caikit Library like".format(
                    ext_name
                ),
            )
