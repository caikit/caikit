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
Shared registry of known backend types
"""

# Standard
from typing import Any, Dict, Optional, Type

# First Party
import alog

# Local
from ..toolkit.errors import error_handler
from ..toolkit.wip_decorator import Action, WipCategory, work_in_progress
from .base import BackendBase

log = alog.use_channel("BCKENDTYP")
error = error_handler.get(log)

## Backend Type Extensible Enum ################################################


class _AttrAccessDict(dict):
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
MODULE_BACKEND_TYPES = _AttrAccessDict()
MODULE_BACKEND_CONFIG_FUNCTIONS: Dict[int, Type[BackendBase]] = {}


## Public ######################################################################


@work_in_progress(action=Action.WARNING, category=WipCategory.BETA)
def register_backend_type(config_class: Optional[Type[BackendBase]] = None):
    """Register a new module backend type by name. This will perform the
    registration if the value does not already exist in the shared enum and will
    be a no-op if it is already present. The backend name should be present as
    a property of the config_class

    Args:
        config_class: BackendBase
            Module to configure particular backend
    """
    # Type validation
    error.type_check(
        "<COR72253432E>",
        type(BackendBase),
        allow_none=True,
        config_class=config_class,
    )

    # NOTE: if a config_class is of `BackendBase` then it will always have
    # backend_type as property which contains the name
    type_name = config_class.backend_type

    error.type_check("<COR92253963E>", str, type_name=type_name)

    # Case check
    error.value_check(
        "<COR07825070E>",
        type_name.upper() == type_name,
        "type_name must be fully UPPERCASE: {}",
        type_name,
    )

    # Add to the global registries
    # NOTE: This only contains a module name and does not contain an object of the backend module
    # The object of the "configured" backend module is generated using caikit.config.configure
    if type_name not in MODULE_BACKEND_TYPES:
        MODULE_BACKEND_TYPES[type_name] = type_name
        MODULE_BACKEND_CONFIG_FUNCTIONS[type_name] = config_class


def __getattr__(name):
    """This module forwards attribute access to the MODULE_BACKEND_TYPES
    mapping
    """
    if not name.startswith("_"):
        return getattr(MODULE_BACKEND_TYPES, name)
