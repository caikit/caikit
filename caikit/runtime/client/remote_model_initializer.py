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
The RemoteModelInitializer loads a RemoteModuleConfig as an empty Module that
sends all requests to an external runtime server

Configuration for RemoteModelInitializer lives under the config as follows:

model_management:
    initializers:
        <initializer name>:
            type: REMOTE
"""
# Standard
from typing import Optional, Type

# First Party
import aconfig
import alog

# Local
from caikit.core.exceptions import error_handler
from caikit.core.model_management.factories import model_initializer_factory
from caikit.core.model_management.model_initializer_base import ModelInitializerBase
from caikit.core.modules import ModuleBase
from caikit.runtime.client.remote_config import RemoteModuleConfig
from caikit.runtime.client.remote_module_base import construct_remote_module_class

log = alog.use_channel("RINIT")
error = error_handler.get(log)


class RemoteModelInitializer(ModelInitializerBase):
    __doc__ = __doc__
    name = "REMOTE"

    def __init__(self, config: aconfig.Config, instance_name: str):
        """Construct with the config"""
        self._instance_name = instance_name
        self._module_class_map = {}

    def init(self, model_config: RemoteModuleConfig, **kwargs) -> Optional[ModuleBase]:
        """Given a RemoteModuleConfig, initialize a RemoteModule instance"""

        # Ensure the module config was produced by a RemoteModelFinder
        if not isinstance(model_config, RemoteModuleConfig):
            log.debug(
                "Initializer %s only supports RemoteModuleConfigs", self._instance_name
            )
            return

        # Construct remote module class if one has not already been created
        self._module_class_map.setdefault(
            model_config.module_id,
            self.construct_module_class(model_config=model_config),
        )

        remote_module_class = self._module_class_map[model_config.module_id]
        return remote_module_class(
            model_config.connection,
            model_config.protocol,
            model_config.model_key,
            model_config.model_path,
        )

    def construct_module_class(
        self, model_config: RemoteModuleConfig
    ) -> Type[ModuleBase]:
        """Helper function to construct a ModuleClass. This is a separate function to allow
         for easy overloading

         Args:
             model_config: RemoteModuleConfig
                The model config to construct the module from

        Returns:
            module: Type[ModuleBase]
                The constructed module"""
        return construct_remote_module_class(model_config)


# Register the remote finder once it has been constructed
model_initializer_factory.register(RemoteModelInitializer)
