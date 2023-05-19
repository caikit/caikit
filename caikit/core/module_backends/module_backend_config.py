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
from typing import List
import copy

# First Party
import alog

# Local
from ..registries import (
    module_backend_classes,
    module_backend_registry,
    module_backend_types,
)
from ..toolkit.errors import error_handler
from .base import BackendBase
from caikit.config import get_config

log = alog.use_channel("CONF")
error = error_handler.get(log)


def start_backends() -> None:
    """This function kicks off the `start` functions for configured backends
    Returns:
        None
    """
    for backend in _CONFIGURED_LOAD_BACKENDS:
        with backend.start_lock():
            if not backend.is_started:
                backend.start()
    for backend in _CONFIGURED_TRAIN_BACKENDS:
        with backend.start_lock():
            if not backend.is_started:
                backend.start()


def configured_load_backends() -> List[BackendBase]:
    """This function returns the list of configured load backends"""
    return copy.copy(_CONFIGURED_LOAD_BACKENDS)


def configured_train_backends() -> List[BackendBase]:
    """This function returns the list of configured train backends"""
    return copy.copy(_CONFIGURED_TRAIN_BACKENDS)


def configure():
    """Configure the backend environment

    NOTE: This function is NOT thread safe!
    """
    config_object = get_config().module_backends
    log.debug3("Full Config: %s", config_object)

    # Configure both train and load backends
    for backend_priority, registry, registry_name in [
        (config_object.load_priority, _CONFIGURED_LOAD_BACKENDS, "load"),
        (config_object.train_priority, _CONFIGURED_TRAIN_BACKENDS, "train"),
    ]:
        backend_priority = backend_priority or []
        error.type_check("<COR46006487E>", list, backend_priority=backend_priority)

        # Configure each backend instance
        for i, backend_config in enumerate(backend_priority):
            error.value_check(
                "<COR48633635E>",
                "type" in backend_config,
                "All backend priority configs must have a 'type' field",
            )
            backend_type = backend_config.type
            error.value_check(
                "<COR72281596E>",
                backend_type in module_backend_types(),
                "Invalid backend [{}] found at backend_priority index [{}]",
                backend_type,
                i,
            )

            log.debug("Configuring backend (%d)[%s]", i, backend_type)
            backend_instance_config = backend_config.get("config", {})
            log.debug3(
                "Backend (%d)[%s] config: %s", i, backend_type, backend_instance_config
            )

            backend_class = module_backend_classes().get(backend_type)
            error.value_check(
                "<COR64618509E>",
                len(registry) == i,
                "({})[{}] already configured",
                i,
                backend_type,
            )
            error.value_check(
                "<COR39517372E>",
                backend_class is not None,
                "Unsupported backend type {}",
                backend_type,
            )
            if not isinstance(backend_class, type) and issubclass(
                backend_class, BackendBase
            ):
                error(
                    "<COR05184600E>",
                    TypeError(
                        f"Backend {backend_class} is not derived from BackendBase"
                    ),
                )

            log.debug2("Performing config for (%d)[%s]", i, backend_type)
            backend_instance = backend_class(backend_instance_config)

            # Add configuration to backends as per individual module requirements
            _configure_backend_overrides(backend_type, backend_instance)

            # Add the instance to the registry
            registry.append(backend_instance)

        log.debug2("All configured %s backends: %s", registry_name, registry)


## Implementation Details ######################################################

# Singleton registries for load and train backends in priority order
_CONFIGURED_LOAD_BACKENDS = []
_CONFIGURED_TRAIN_BACKENDS = []


def _configure_backend_overrides(backend: str, backend_instance: object):
    """Function to go over all the modules registered in the MODULE_BACKEND_REGISTRY
    for a particular backend and configure their backend overrides

    Args:
        backend: str
            Name of the backend to select from registry
        backend_instance: object
            Initialized backend instance. This object should
            implement the `register_config` function which will be
            used to merge / iteratively configure the backend
    """
    # Go through all the modules registered with particular backend
    for module_id, module_type_mapping in module_backend_registry().items():
        if backend in module_type_mapping:
            # check if it contains any special config
            config = module_type_mapping[backend].backend_config_override
            error.type_check("<COR61136899E>", dict, config=config)
            if len(config) != 0:
                # TODO: Add a check here to see if the backend has already started
                backend_instance.register_config(config)
            else:
                log.debug2(
                    f"No backend overrides configured for {module_id} module and {backend} backend"
                )
