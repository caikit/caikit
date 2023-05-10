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
import threading

# First Party
import aconfig
import alog

# Local
from .module import MODULE_BACKEND_REGISTRY
from .module_backends.backend_types import (
    MODULE_BACKEND_CONFIG_FUNCTIONS,
    MODULE_BACKEND_TYPES,
)
from .module_backends.base import BackendBase
from .toolkit.errors import error_handler
from caikit.config import get_config

log = alog.use_channel("CONF")
error = error_handler.get(log)


def start_backends() -> None:
    """This function kicks off the `start` functions for configured backends
    Returns:
        None
    """
    for backend in _CONFIGURED_LOAD_BACKENDS:
        with _BACKEND_START_LOCKS[backend.name]:
            if not backend.is_started:
                backend.start()
    for backend in _CONFIGURED_TRAIN_BACKENDS:
        with _BACKEND_START_LOCKS[backend.name]:
            if not backend.is_started:
                backend.start()


def get_load_backend(backend_name: str) -> BackendBase:
    """Get the configured instance of the given backend type. If not configured,
    a ValueError is raised
    """
    return _get_registry_backend(backend_name, "load", _CONFIGURED_LOAD_BACKENDS)


def get_train_backend(backend_name: str) -> BackendBase:
    """Get the configured instance of the given backend type. If not configured,
    a ValueError is raised
    """
    return _get_registry_backend(backend_name, "train", _CONFIGURED_TRAIN_BACKENDS)


def configured_load_backends() -> List[BackendBase]:
    """This function returns the mapping of named"""
    return copy.copy(_CONFIGURED_LOAD_BACKENDS)


def configured_train_backends() -> List[BackendBase]:
    """This function exposes the list of configured train backends for downstream
    checks
    """
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

        # Check if disable_local is set
        disable_local_backend = config_object.disable_local or False

        # Add local at the end of priority by default
        backend_priority_types = [cfg.get("type") for cfg in backend_priority]
        error.value_check(
            "<COR92038969E>",
            not (
                disable_local_backend
                and MODULE_BACKEND_TYPES.LOCAL in backend_priority_types
            ),
            "Invalid configuration with {} in the priority list and disable_local set",
            MODULE_BACKEND_TYPES.LOCAL,
        )
        if not disable_local_backend and (
            MODULE_BACKEND_TYPES.LOCAL not in backend_priority_types
        ):
            log.debug3("Adding fallback priority to [%s]", MODULE_BACKEND_TYPES.LOCAL)
            backend_priority.append(
                aconfig.Config(
                    {"type": MODULE_BACKEND_TYPES.LOCAL},
                    override_env_vars=False,
                )
            )

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
                backend_type in MODULE_BACKEND_TYPES,
                "Invalid backend [{}] found at backend_priority index [{}]",
                backend_type,
                i,
            )

            backend_name = backend_config.get("name", backend_type)
            log.debug("Configuring backend [%s]", backend_name)
            backend_instance_config = backend_config.get("config", {})
            log.debug3("Backend [%s] config: %s", backend_name, backend_instance_config)

            backend_class = MODULE_BACKEND_CONFIG_FUNCTIONS.get(backend_type)
            error.value_check(
                "<COR64618509E>",
                not any(
                    backend.name == backend_name
                    and backend.backend_type == backend_type
                    for backend in registry
                ),
                "{}/{} already configured",
                backend_name,
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

            log.debug2("Performing config for [%s]", backend_name)
            backend_instance = backend_class(backend_name, backend_instance_config)

            # Add configuration to backends as per individual module requirements
            _configure_backend_overrides(backend_name, backend_instance)

            # NOTE: Configured backends holds the object of backend classes that are based
            # on BaseBackend
            # The start operation of the backend needs to be performed separately
            registry.append(backend_instance)
            _BACKEND_START_LOCKS[backend_name] = threading.Lock()

        log.debug2("All configured %s backends: %s", registry_name, registry)


## Implementation Details ######################################################

# Singleton registries for load and train backends in priority order
_CONFIGURED_LOAD_BACKENDS = []
_CONFIGURED_TRAIN_BACKENDS = []

# Locks for starting backends
# NOTE: A single lock is held created for a backend name, even if it is repeated
#   between train and load. Configuration is a load-time operation, so while
#   this might be slightly overly locking, it protects against the case where
#   a train and load backend have overlapping global state.
_BACKEND_START_LOCKS = {}


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
    for module_id, module_type_mapping in MODULE_BACKEND_REGISTRY.items():
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


def _get_registry_backend(
    backend_name: str, registry_name: str, registry: list
) -> BackendBase:
    """Get the configured instance of the given backend type. If not configured,
    a ValueError is raised
    """
    matching_backends = [
        backend for backend in registry if backend.name == backend_name
    ]
    error.value_check(
        "<COR82987857E>",
        matching_backends,
        "Cannot fetch unconfigured {} backend [{}]",
        registry_name,
        backend_name,
    )
    assert (
        len(matching_backends) == 1
    ), "PROGRAMMING ERROR: Duplicate names should be prohibited"
    backend = matching_backends[0]
    if not backend.is_started:
        with _BACKEND_START_LOCKS[backend_name]:
            if not backend.is_started:
                backend.start()
    return backend
