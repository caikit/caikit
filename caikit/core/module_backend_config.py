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
from typing import List, Optional
import os
import threading

# First Party
# First party
import aconfig
import alog

# Local
from .module import MODULE_BACKEND_REGISTRY
from .module_backends.backend_types import (
    MODULE_BACKEND_CONFIG_FUNCTIONS,
    MODULE_BACKEND_TYPES,
)
from .module_backends.base import BackendBase
from .toolkit.config_utils import merge_configs
from .toolkit.errors import error_handler

log = alog.use_channel("CONF")
error = error_handler.get(log)

_CONFIGURED_BACKENDS = {}
_BACKEND_START_LOCKS = {}


def start_backends() -> None:
    """This function kicks off the `start` functions for configured backends
    Returns:
        None
    """
    for backend_type, backend in _CONFIGURED_BACKENDS.items():
        with _BACKEND_START_LOCKS[backend_type]:
            if not backend.is_started:
                backend.start()


def get_backend(backend_type: str) -> BackendBase:
    """Get the configured instance of the given backend type. If not configured,
    a ValueError is raised
    """
    error.value_check(
        "<COR82987857E>",
        backend_type in _CONFIGURED_BACKENDS,
        "Cannot fetch unconfigured backend [%s]",
        backend_type,
    )
    backend = _CONFIGURED_BACKENDS[backend_type]
    if not backend.is_started:
        with _BACKEND_START_LOCKS[backend_type]:
            if not backend.is_started:
                backend.start()
    return backend


def configured_backends() -> List[str]:
    """This function exposes the list of configured backends for downstream
    checks
    """
    # NOTE: Return a copy to avoid accidentally mutating the global
    return list(_CONFIGURED_BACKENDS.keys())


# TODO: These keys esp. backend priority need update to work with new config
def configure(*_, config_file: Optional[str] = None, **overrides):
    """Configure the backend environment based on configuration available
    in the given arguments.

    NOTE: This function is NOT thread safe!

    Kwargs:
        config_file:  Optional[str]
            Path to a configuration yaml file to use instead of the default
        **overrides
            overrides dict to apply on top of the loaded coniguration
    """
    # Load the config file either from the default location or from the given
    # path with environment overides
    config_file = config_file or _DEFAULT_CONFIG_FILE
    error.file_check("<COR37636675E>", config_file)

    config_object = aconfig.Config.from_yaml(config_file, override_env_vars=True)

    # Deep merge the given overrides on top of the the loaded config
    config_object = merge_configs(config_object, overrides)

    log.debug3("Full Config: %s", config_object)

    # Determine the priority list of enabled backends
    #
    # NOTE: All backends are held in UPPERCASE, but this is not canonical for
    #   yaml or function arguments, so we allow lowercase names in the config
    #   and coerce them to upper here
    backend_priority = config_object.backend_priority or []
    error.type_check("<COR46006487E>", list, backend_priority=backend_priority)

    # Check if disable_local_backend is set
    disable_local_backend = config_object.disable_local_backend or False

    # Add local at the end of priority by default
    # TODO: Should we remove LOCAL from priority if it is disabled?
    if not disable_local_backend and (
        MODULE_BACKEND_TYPES.LOCAL not in backend_priority
    ):
        log.debug3("Adding fallback priority to [%s]", MODULE_BACKEND_TYPES.LOCAL)
        backend_priority.append(MODULE_BACKEND_TYPES.LOCAL)
    backend_priority = [backend.upper() for backend in backend_priority]

    for i, backend in enumerate(backend_priority):
        error.value_check(
            "<COR72281596E>",
            backend in MODULE_BACKEND_TYPES,
            "Invalid backend [{}] found at backend_priority index [{}]",
            backend,
            i,
        )
    log.debug2("Backend Priority: %s", backend_priority)

    # Iterate through the config objects for each enabled backend in order and
    # do the actual config
    backend_configs = {
        key.lower(): val for key, val in config_object.get("backends", {}).items()
    }
    for backend in backend_priority:
        log.debug("Configuring backend [%s]", backend)
        backend_config = backend_configs.get(backend.lower(), {})
        log.debug3("Backend [%s] config: %s", backend, backend_config)

        if backend in configured_backends() and backend != MODULE_BACKEND_TYPES.LOCAL:
            error("<COR64618509E>", AssertionError(f"{backend} already configured"))

        config_class = MODULE_BACKEND_CONFIG_FUNCTIONS.get(backend)

        # NOTE: since all backends needs to be derived from BackendBase, they all
        # support configuration. as input
        if config_class is not None:
            log.debug2("Performing config for [%s]", backend)
            config_class_obj = config_class(backend_config)

            # Add configuration to backends as per individual module requirements
            _configure_backend_overrides(backend, config_class_obj)

        # NOTE: Configured backends holds the object of backend classes that are based
        # on BaseBackend
        # The start operation of the backend needs to be performed separately
        _CONFIGURED_BACKENDS[backend] = config_class_obj
        _BACKEND_START_LOCKS[backend] = threading.Lock()

    log.debug2("All configured backends: %s", _CONFIGURED_BACKENDS)


## Implementation Details ######################################################

_DEFAULT_CONFIG_FILE = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "config", "config.yml")
)

# The global map of configured backends
_CONFIGURED_BACKENDS = {}


# Thread-safe lock for each backend to ensure that starting is not performed
# multiple times on any given backend
_BACKEND_START_LOCKS = {}


def _configure_backend_overrides(backend: str, config_class_obj: object):
    """Function to go over all the modules registered in the MODULE_BACKEND_REGISTRY
    for a particular backend and configure their backend overrides

    Args:
        backend: str
            Name of the backend to select from registry
        config_class_obj: object
            Initialized object of Backend module. This object should
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
                config_class_obj.register_config(config)
            else:
                log.debug2(
                    f"No backend overrides configured for {module_id} module and {backend} backend"
                )
