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
The LocalModelLoader loads a model into local memory
"""
# Standard
from typing import Optional, Union
import os

# First Party
import aconfig
import alog

# Local
from ..module_backends import backend_types, module_backend_config
from ..module_backends.module_backend_config import (  # TODO: Move here!
    _configure_backend_overrides,
)
from ..modules import ModuleBase, ModuleConfig
from ..modules.decorator import SUPPORTED_LOAD_BACKENDS_VAR_NAME
from ..registries import (
    module_backend_classes,
    module_backend_registry,
    module_backend_types,
    module_registry,
)
from ..toolkit.errors import error_handler
from .model_loader_base import ModelLoaderBase

log = alog.use_channel("LLOAD")
error = error_handler.get(log)


class LocalModelLoader(ModelLoaderBase):
    __doc__ = __doc__

    name = "LOCAL"

    def __init__(self, config: aconfig.Config):
        """Construct with the config"""
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
            _configure_backend_overrides(backend_type, backend_instance)

            # Add the instance to the registry
            self._backends.append(backend_instance)

        log.debug2("All configured backends: %s", self._backends)

    def load(
        self, model_config: ModuleConfig, **kwargs
    ) -> Union[Optional[ModuleBase], Exception]:
        """Given a ModelConfig, attempt to load it into memory

        Args:
            model_config (ModuleConfig): The in-memory model config object for
                the model to be loaded

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
                loaded_model = module_backend_impl.load(
                    model_path,
                    load_backend=load_backend,
                    **kwargs,
                )
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
