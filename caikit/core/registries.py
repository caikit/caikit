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
"""
Single spot for all global shared state access

This should not import anything else from caikit outside the toolkit error handler, many parts of
caikit need to access this state so that would easily cause an import cycle.
"""

# Standard
from typing import Any, Dict, Tuple, Type

# First Party
import alog

# Local
from .toolkit.errors import error_handler

log = alog.use_channel("REGISTRIES")
error = error_handler.get(log)


# Registry of all classes decorated with `@caikit.module`
MODULE_REGISTRY = {}

# Global module backend registry dict
MODULE_BACKEND_REGISTRY = {}


def module_registry() -> Dict[str, "caikit.core.ModuleBase"]:
    """🌶️🌶️🌶️ This returns global state that should only be mutated if you know what you're doing!

    Returns the dictionary of decorated @modules that have been imported.
    Used to map module IDs to the concrete class implementations to load and run.

    Structure is
    Dict[ module_id, module_class ]

    Returns:
        Dict[str, caikit.core.ModuleBase]: The module registry
    """
    return MODULE_REGISTRY


def module_backend_registry() -> Dict[
    str, Dict[str, Tuple["caikit.core.ModuleBase", Dict]]
]:
    """🌶️🌶️🌶️ This returns global state that should only be mutated if you know what you're doing!

    Returns the module backend registry. This adds more nesting to the module registry,
    providing a dictionary of backend type name -> (backend module impl class, config dict)

    Structure is
    Dict[ module_id, Dict[ backend_type, Tuple[ backend_impl_class, backend_config_dict ] ] ]

    Returns:
        Dict[str, Dict[str, Tuple["caikit.core.BackendBase", Dict]]]: The module
            backend registry
    """
    # TODO: put a real data structure here instead of nested dicts
    return MODULE_BACKEND_REGISTRY


class _AttrAccessBackendDict(dict):
    """Simple extension on a dict that allows attribute access in addition to
    index lookup
    """

    def __getattr__(self, name: str) -> Any:
        """Alias to index lookup"""
        error.value_check(
            "<COR85015051E>", name in self, "backend type {} not registered", name
        )
        return self[name]


# "enum" holding known backend types. This is implemented as a dict so that it
# can be extended as needed by downstream libraries.
MODULE_BACKEND_TYPES = _AttrAccessBackendDict()


def module_backend_types() -> Dict[str, str]:
    """🌶️🌶️🌶️ This returns global state that should only be mutated if you know what you're doing!

    Returns the "enum" of module backend types. This is a dict where the keys and values
    are identical, and each are the string names of a backend type.

    Returns:
        Dict[str, str]:
            The module backend type enum
    """
    return MODULE_BACKEND_TYPES


MODULE_BACKEND_CLASSES: Dict[str, Type["caikit.core.BackendBase"]] = {}


def module_backend_classes() -> Dict[str, Type["caikit.core.BackendBase"]]:
    """🌶️🌶️🌶️ This returns global state that should only be mutated if you know what you're doing!

    Returns the mapping of backend type name to concrete backend class

    Returns:
        Dict[str, Type["caikit.core.BackendBase"]]:
            The module backend class registry
    """

    return MODULE_BACKEND_CLASSES
