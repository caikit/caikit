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
Single spot where stick accessors for global shared state

(Shining the light on the shared state so that we can clean it up and re-home it)
This SHOULD NOT import anything else from caikit outside the toolkit error handler
"""

# Standard
from typing import Any, Dict, Tuple, Type

# First Party
import alog

# Local
from caikit.core.toolkit.errors import error_handler

log = alog.use_channel("REGISTRIES")
error = error_handler.get(log)


# Registry of all classes decorated with `@caikit.module`
MODULE_REGISTRY = {}

# Global module backend registry dict
MODULE_BACKEND_REGISTRY = {}


def module_registry() -> Dict[str, "caikit.core.ModuleBase"]:
    return MODULE_REGISTRY


def module_backend_registry() -> Dict[str, Dict["caikit.core.BackendBase", Tuple]]:
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


def module_backend_types() -> Dict[Dict[str, str]]:
    return MODULE_BACKEND_TYPES


MODULE_BACKEND_CLASSES: Dict[str, Type["caikit.core.BackendBase"]] = {}


def module_backend_classes() -> Dict[str, Type["caikit.core.BackendBase"]]:
    return MODULE_BACKEND_CLASSES
