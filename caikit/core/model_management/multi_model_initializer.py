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
The MultiModelInitializer configures a set of other model initializers that will be used
in sequence to try loading models.

Configuration for MultiModelInitializer lives under the config as follows:

model_management:
    initializers:
        <initializer name>:
            type: MULTI
            config:
                # Sequence of other initializer names to use in priority order
                initializer_priority:
                    - other_initializer1
                    - other_initializer2
"""
# Standard
from typing import Optional

# First Party
import aconfig
import alog

# Local
from ...config import get_config
from ..exceptions import error_handler
from ..modules import ModuleBase, ModuleConfig
from .model_initializer_base import ModelInitializerBase

# NOTE: Top-level import done so that global MODEL_MANAGER can be used at
#   construction time without incurring a circular dependency
import caikit.core

log = alog.use_channel("MINIT")
error = error_handler.get(log)


class MultiModelInitializer(ModelInitializerBase):
    __doc__ = __doc__

    name = "MULTI"

    def __init__(self, config: aconfig.Config, instance_name: str):
        """Initialize with the sequence of initializers to use"""
        self._instance_name = instance_name
        initializer_priority = config.initializer_priority
        error.type_check(
            "<COR47518221E>",
            list,
            initializer_priority=initializer_priority,
        )
        error.type_check_all(
            "<COR47518222E>",
            str,
            initializer_priority=initializer_priority,
        )
        error.value_check(
            "<COR05042343E>",
            initializer_priority,
            "Must provide at least one valid initializer",
        )
        config_initializers = get_config().model_management.initializers
        invalid_initializers = [
            initializer
            for initializer in initializer_priority
            if initializer not in config_initializers
        ]
        error.value_check(
            "<COR68252034E>",
            not invalid_initializers,
            "Invalid initializers given in initializer_priority: {}",
            invalid_initializers,
        )
        error.value_check(
            "<COR54613971E>",
            self._instance_name not in initializer_priority,
            "Cannot include self in multi initializer priority",
        )
        model_manager = config.model_manager or caikit.core.MODEL_MANAGER
        log.debug2(
            "Setting up %s with initializer priority: %s",
            self.name,
            initializer_priority,
        )
        self._initializers = [
            model_manager.get_initializer(initializer)
            for initializer in initializer_priority
        ]

    def init(
        self,
        model_config: ModuleConfig,
        **kwargs,
    ) -> Optional[ModuleBase]:
        """Iterate through the sequence of initializers and return the first one that
        succeeds
        """
        for idx, initializer in enumerate(self._initializers):
            log.debug2(
                "Trying to init %s with initializer %d of type %s",
                model_config.module_id,
                idx,
                initializer.name,
            )
            try:
                module = initializer.init(model_config, **kwargs)
                if module:
                    log.debug(
                        "Init model %s with initializer %d of type %s",
                        model_config.module_id,
                        idx,
                        initializer.name,
                    )
                    return module
                log.debug2(
                    "Initializer %d of type %s unable to init %s",
                    idx,
                    initializer.name,
                    model_config.module_id,
                )
            except Exception as err:  # pylint: disable=broad-exception-caught
                log.debug2(
                    "Initializer %d of type %s failed to load %s: %s",
                    idx,
                    initializer.name,
                    model_config.module_id,
                    err,
                )
                log.debug4("Initializer error", exc_info=True)

        # No initializer succeeded
        log.warning("Unable to init %s with any initializer", model_config.module_id)
        return None
