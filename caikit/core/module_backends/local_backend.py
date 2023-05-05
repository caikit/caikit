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

"""This module implements a LOCAL backend configuration
"""
# Standard
from typing import Type

# First Party
import alog

# Local
from ..module import MODULE_REGISTRY, ModuleBase
from ..toolkit.errors import error_handler
from .backend_types import register_backend_type
from .base import UniversalLoadBackendBase, UniversalTrainBackendBase

log = alog.use_channel("LCLBKND")
error = error_handler.get(log)


class LocalBackend(UniversalLoadBackendBase, UniversalTrainBackendBase):
    backend_type = "LOCAL"

    def register_config(self, config) -> None:
        """Function to merge configs with existing configurations"""
        error(
            "<COR86557945E>",
            AssertionError(
                f"{self.backend_type} backend does not support this operation"
            ),
        )

    def start(self):
        """Start local backend. This is a no-op function"""
        self._started = True

    def stop(self):
        """Stop local backend. This is a no-op"""
        self._started = False

    def train(self, module_class: Type[ModuleBase], *args, **kwargs) -> ModuleBase:
        """Perform a local training on the given class"""
        with alog.ContextTimer(log.info, "Finished local training in: "):
            return module_class.train(*args, **kwargs)

    def load(self, module_id: str, model_path: str, *args, **kwargs) -> ModuleBase:
        """Look up the given module in the module registry and load it if found"""
        module_class = MODULE_REGISTRY.get(module_id)
        if module_class is not None:
            return module_class.load(model_path, *args, **kwargs)


# Register local backend
register_backend_type(LocalBackend)
