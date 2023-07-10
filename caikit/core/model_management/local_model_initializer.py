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
The LocalModelInitializer loads a model locally, optionally with a non-local backend
if the given module provides multiple backend-specific implementations.

Configuration for LocalModelInitializer lives under the config as follows:

model_management:
    initializers:
        <initializer name>:
            type: LOCAL
            config:
                # List of module backend configurations in priority order
                backend_priority:
                    - type: LOCAL
                      config: {}
"""
# Standard
from typing import Callable, Optional
import inspect

# First Party
import aconfig
import alog

# Local
from ..module_backends import BackendBase, backend_types
from ..modules import ModuleBase, ModuleConfig
from ..modules.decorator import SUPPORTED_LOAD_BACKENDS_VAR_NAME
from ..registries import (
    module_backend_classes,
    module_backend_registry,
    module_backend_types,
)
from ..toolkit.errors import error_handler
from .model_initializer_base import ModelInitializerBase

log = alog.use_channel("LLOAD")
error = error_handler.get(log)


class LocalModelInitializer(ModelInitializerBase):
    __doc__ = __doc__

    name = "LOCAL"

    def __init__(self, config: aconfig.Config, instance_name: str):
        """Construct with the config"""
        self._instance_name = instance_name
        self._backends = []
        backend_priority = config.get(
            "backend_priority",
            [aconfig.Config({"type": "LOCAL"}, override_env_vars=False)],
        )
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
                len(self._backends) == i,
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
            self._configure_backend_overrides(backend_type, backend_instance)

            # Add the instance to the registry
            self._backends.append(backend_instance)

        log.debug2(
            "All configured backends for %s: %s", self._instance_name, self._backends
        )

    def init(self, model_config: ModuleConfig, **kwargs) -> Optional[ModuleBase]:
        """Given a ModelConfig, attempt to initialize it locally, possibly using
        a non-local backend

        Args:
            model_config (ModuleConfig): The in-memory model config object for
                the model to be initialized

        Returns:
            model (Optional[ModuleBase]): The in-memory ModuleBase instance that
                is ready to run
        """
        module_id = model_config.module_id
        model_path = model_config.model_path
        module_implementations = module_backend_registry().get(module_id, {})
        log.debug2(
            "Number of available backend implementations for %s found: %d",
            module_id,
            len(module_implementations),
        )
        # Look up the backend that this model was created with
        model_creation_backend = model_config.get("model_backend", backend_types.LOCAL)
        log.debug2("Model creation backend: %s", model_creation_backend)

        # Iterate through each backend in priority order and see if this module
        # can load with it
        loaded_model = None
        for load_backend in self._backends:

            # Look in the module's implementations for this backend type
            backend_impl_obj = module_implementations.get(load_backend.backend_type)
            if backend_impl_obj is None:
                log.debug3(
                    "Module %s does not support loading with %s",
                    module_id,
                    load_backend.backend_type,
                )
                continue

            # Grab the concrete module class for this backend and check to
            # see if this model's artifacts were created with a version of
            # the module that can be loaded with this backend.
            module_backend_impl = backend_impl_obj.impl_class
            supported_load_backends = self._get_supported_load_backends(
                module_backend_impl
            )
            if model_creation_backend in supported_load_backends:
                log.debug3(
                    "Attempting to load %s (module_id %s) with backend %s and class %s",
                    model_path,
                    module_id,
                    load_backend.backend_type,
                    module_backend_impl.__name__,
                )
                extra_kwargs = {}
                if self._supports_load_backend_kwarg(module_backend_impl.load):
                    extra_kwargs["load_backend"] = load_backend
                loaded_model = module_backend_impl.load(
                    model_path,
                    **extra_kwargs,
                    **kwargs,
                )
                error.type_check("<COR40080753E>", ModuleBase, model=loaded_model)
                if loaded_model is not None:
                    log.debug2(
                        "Successfully loaded %s with backend %s",
                        model_path,
                        load_backend.backend_type,
                    )
                    loaded_model.set_load_backend(load_backend)
                    break

        # Return the loaded model if it was able to load
        return loaded_model

    ## Implementation Details ##################################################

    @staticmethod
    def _supports_load_backend_kwarg(load_fn: Callable) -> bool:
        """A load function supports the load_backend kwarg IFF it has an arg
        explicitly named load_backend or it has a ** kwarg capture
        """
        sig = inspect.signature(load_fn)
        return "load_backend" in sig.parameters

    def _get_supported_load_backends(self, backend_impl: ModuleBase):
        """Function to get a list of supported load backends
        that the module supports

        Args:
            backend_impl: caikit.core.ModuleBase
                Module implementing the backend
        Returns:
            list(backend_types)
                list of backends that are supported for model load
        """

        # Get list of backends that are supported for load
        # NOTE: since code in a module can change anytime, its support
        # for various backend might also change, in which case,
        # it would be better to keep the backend information in the model itself
        # If module_backend is None, then we will assume that this model is not loadable in
        # any other backend
        return getattr(backend_impl, SUPPORTED_LOAD_BACKENDS_VAR_NAME, [])

    @staticmethod
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
                        "No backend overrides configured for %s module and %s backend",
                        module_id,
                        backend,
                    )
