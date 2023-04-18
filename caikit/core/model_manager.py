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


"""Most logic interacting with models.  Can load, etc.
"""

# Standard
from io import BytesIO
from typing import Union
import os
import tempfile
import zipfile

# First Party
import alog

# Local
from . import module_backend_config as backend_config
from .config import lib_config
from .module import (
    _MODULE_TYPES,
    MODULE_BACKEND_REGISTRY,
    MODULE_REGISTRY,
    SUPPORTED_LOAD_BACKENDS_VAR_NAME,
    ModuleBase,
    ModuleConfig,
)
from .module_backends import backend_types
from .toolkit.errors import error_handler

log = alog.use_channel("MDLMNG")
error = error_handler.get(log)

# restrict functions that are imported so we don't pollute the base module namespace
__all__ = [
    "get_valid_module_ids",
    "ModelManager",
]


def get_valid_module_ids():
    """Get a dictionary mapping all module (block and workflow) IDs to the
    string names of the implementing classes.
    """
    return {
        module_id: model_class.__name__
        for module_id, model_class in MODULE_REGISTRY.items()
    }


class ModelManager:
    """Manage the models or resources for library."""

    def __init__(self):
        """Initialize ModelManager."""
        # Map to store module caches, to be used for singleton model lookups
        self.singleton_module_cache = {}

    # make load function available from top-level of library
    def load(self, module_path, *args, load_singleton=False, **kwargs):
        """Load a model and return an instantiated object on which we can run inference.

        Args:
            module_path: str | BytesIO | bytes
                A module path to one of the following.
                    1. A module path to a directory containing a yaml config file in the top level.
                    2. A module path to a zip archive containing either a yaml config file in the
                       top level when extracted, or a directory containing a yaml config file in
                       the top level.
                    3. A BytesIO object corresponding to a zip archive containing either a yaml
                       config file in the top level when extracted, or a directory containing a
                       yaml config file in the top level.
                    4. A bytes object corresponding to a zip archive containing either a yaml
                       config file in the top level when extracted, or a directory containing a
                       yaml config file in the top level.
            load_singleton: bool (Defaults to False)
                Indicates whether this model should be loaded as a singleton.

        Returns:
            subclass of blocks.base.BlockBase
                Model object that is loaded, configured, and ready for prediction.
        """
        error.type_check("<COR98255724E>", bool, load_singleton=load_singleton)

        # This allows a user to load their own model (e.g. model saved to disk)
        load_path = lib_config.load_path
        if load_path is not None and isinstance(module_path, str):
            if not os.path.exists(module_path):
                module_path = os.path.join(load_path, module_path)

        # Ensure that we have a loadable directory.
        error.type_check("<COR98255419E>", str, BytesIO, bytes, module_path=module_path)
        if isinstance(module_path, str):
            # Ensure this path is operating system correct if it isn't already.
            module_path = os.path.normpath(module_path)
        # If we have bytes, convert to a buffer, since we already handle in memory binary streams.
        elif isinstance(module_path, bytes):
            module_path = BytesIO(module_path)
        # Now that we have a file like object | str we can try to load as an archive.
        if zipfile.is_zipfile(module_path):
            return self._load_from_zipfile(module_path, load_singleton, *args, **kwargs)
        try:
            return self._load_from_dir(module_path, load_singleton, *args, **kwargs)
        except FileNotFoundError:
            error(
                "<COR80419785E>",
                FileNotFoundError(
                    "Module load path `{}` does not contain a `config.yml` file.".format(
                        module_path
                    )
                ),
            )

    def _load_from_dir(self, module_path, load_singleton, *args, **kwargs):
        """Load a model from a directory.

        Args:
            module_path:  str
                Path to directory. At the top level of directory is `config.yml` which holds info
                about the model.
            load_singleton: bool
                Indicates whether this model should be loaded as a singleton.

        Returns:
            subclass of blocks.base.BlockBase
                Model object that is loaded, configured, and ready for prediction.
        """
        module_config = ModuleConfig.load(module_path)

        # Check if the model is being loaded in singleton fashion. If so,
        # then fetch they hash for the module from config and use it as key.
        key = module_config.unique_hash
        if load_singleton and key is not None and key in self.singleton_module_cache:
            # return model back from singleton cache
            return self.singleton_module_cache[key]

        # retrive and validate the module class to initialize based on the
        # module_id retrieved from the configuration (which is dynamically set
        # based on either a block_id or workflow_id in the config.yml), looking up
        # the corresponding MODULE_ID (BLOCK_ID/WORKFLOW_ID) of the module class
        # in the ModuleBase class registry
        module_id = module_config["module_id"]
        module_class = MODULE_REGISTRY.get(module_id)

        if module_class is None:
            error(
                "<COR50207494E>",
                ValueError(
                    "could not find class with MODULE_ID of `{}`".format(module_id)
                ),
            )

        if not issubclass(module_class, ModuleBase):
            error(
                "<COR18830919E>",
                TypeError(
                    "class `{}` is not a valid module for module load".format(
                        module_class.__name__
                    )
                ),
            )

        ## TODO: Should backend lookup be optional?
        log.debug2("Looking for available backends for module_id %s", module_id)

        # NOTE: module_id is going to be same for all the backends
        # Thus it is not possible to know which backend this model was saved with from the
        # module_id alone. We could look at the *_class field, like `block_class`.
        # but then we have to look at various fields and that also doesn't guarantee the
        # support for those backend in the current code of the backend.

        model_backend = module_config.model_backend or backend_types.LOCAL

        log.debug2("model trained on backend: %s", model_backend)

        # Get the mapping of available implementations for this module
        configured_backends = backend_config.configured_backends()

        if len(configured_backends) == 0:
            log.warning(
                "<COR56759744W>",
                "No backend configured! Trying to configure using default config file.",
            )
            backend_config.configure()
            configured_backends = backend_config.configured_backends()

        # If configured backends is empty, add LOCAL backend to it
        local_enabled = backend_types.LOCAL in configured_backends

        # NOTE: Local backend can be disabled via `config.yml` but its enabled
        # by default

        log.debug2("Local enabled? %s", local_enabled)
        module_implementations = MODULE_BACKEND_REGISTRY.get(
            module_id, {backend_types.LOCAL: None} if local_enabled else {}
        )
        log.debug2(
            "Number of available backend implementations found: %d",
            len(module_implementations),
        )

        error.value_check(
            "<COR84094427E>",
            len(module_implementations) > 0,
            "No implementation of {} available. You may need to `pip install {}`",
            module_config.module_id,
            self.get_module_class_from_config(module_config),
        )

        loaded_artifact = None
        # instantiate object and return to user
        log.debug("Loading the artifact!")

        # Go through each configured backend in priority order and look for an
        # implementation that matches
        # NOTE: backend priority can be configured while configuring backend
        for backend in configured_backends:
            # NOTE: what if we have multiple backends available,
            # this will currently only return 1st one of those
            # based on priority
            backend_impl_obj = module_implementations.get(backend)
            # NOTE: LOCAL may not be marked as a specific implementation, thus skip this
            # or if no implementation available of the provided backend, continue
            if backend == backend_types.LOCAL or backend_impl_obj is None:
                continue

            backend_impl = backend_impl_obj.impl_class

            # A particular model can be supported by multiple backends, i.e
            # a model trained on LOCAL backend might be able to get loaded with
            # Spark or Ray backend implementations of the same block.
            # Here, we will try to get the list of supported load backends for
            # each of the backend_types and check if they contain the
            # backend type of the current model
            supported_load_backends = self._get_supported_load_backends(backend_impl)

            # Check if the module actually supports this backend for load
            if model_backend in supported_load_backends:
                log.debug(
                    "%s backend implementation found for backend [%s]",
                    backend_impl.__name__,
                    backend,
                )
                loaded_artifact: ModuleBase = backend_impl.load(
                    module_path, *args, **kwargs
                )
                break

        # If model not able to load still, it is a model that probably
        # does not define the "backed_type" or "supported_load_backends".
        # These would all be 'LOCAL' backend model, thus try to load with that
        if not loaded_artifact and local_enabled:
            module_class = MODULE_REGISTRY.get(module_id)
            loaded_artifact: ModuleBase = module_class.load(
                module_path, *args, **kwargs
            )

        ### END Backend distribution

        error.value_check(
            "<COR24332812E>",
            loaded_artifact is not None,
            "No available implementation for provided model!",
        )

        # if singleton loading is enabled, and module unique hash is available,
        # save the module in singleton map
        if load_singleton and key is not None:
            self.singleton_module_cache[key] = loaded_artifact

        return loaded_artifact

    def _load_from_zipfile(self, module_path, load_singleton, *args, **kwargs):
        """Load a model from a zip archive.

        Args:
            module_path:  str
                Path to directory. At the top level of directory is `config.yml` which holds info
                about the model.
            load_singleton: bool
                Indicates whether this model should be loaded as a singleton.

        Returns:
            subclass of blocks.base.BlockBase
                Model object that is loaded, configured, and ready for prediction.
        """
        with tempfile.TemporaryDirectory() as extract_path:
            with zipfile.ZipFile(module_path, "r") as zip_f:
                zip_f.extractall(extract_path)
            # Depending on the way the zip archive is packaged, out temp directory may unpack
            # to files directly, or it may unpack to a (single) directory containing the files.
            # We expect the former, but fall back to the second if we can't find the config.
            try:
                model = self._load_from_dir(
                    extract_path, load_singleton, *args, **kwargs
                )
            # NOTE: Error handling is a little gross here, the main reason being that we
            # only want to log to error() if something is fatal, and there are a good amount
            # of things that can go wrong in this process.
            except FileNotFoundError:

                def get_full_path(folder_name):
                    return os.path.join(extract_path, folder_name)

                # Get the contained directories. Omit anything starting with __ to avoid
                # accidentally traversing compression artifacts, e.g., __MACOSX.
                nested_dirs = [
                    get_full_path(f)
                    for f in os.listdir(extract_path)
                    if os.path.isdir(get_full_path(f)) and not f.startswith("__")
                ]
                # If we have multiple dirs, something is probably wrong - this doesn't look
                # like a simple level of nesting as a result of creating the zip.
                if len(nested_dirs) != 1:
                    error(
                        "<COR06761097E>",
                        FileNotFoundError(
                            "Unable to locate archive config due to nested dirs"
                        ),
                    )
                # Otherwise, try again. If we fail again stop, because the zip creation should only
                # create one potential extra layer of nesting around the model directory.
                try:
                    model = self._load_from_dir(
                        nested_dirs[0], load_singleton, *args, **kwargs
                    )
                except FileNotFoundError:
                    error(
                        "<COR84410081E>",
                        FileNotFoundError(
                            "Unable to locate archive config within top two levels of {}".format(
                                module_path
                            )
                        ),
                    )
        return model

    def extract(self, zip_path, model_path, force_overwrite=False):
        """Method to extract a downloaded archive to a specified directory.

        Args:
            zip_path: str
                Location of .zip file to extract.
            model_path: str
                Model directory where the archive should be unzipped unzipped.
            force_overwrite: bool (Defaults to false)
                Force an overwrite to model_path, even if the folder exists
        Returns:
            str
                Output path where the model archive is unzipped.
        """
        model_path = os.path.abspath(model_path)

        # skip if force_overwrite disabled and path already exists
        if not force_overwrite and os.path.exists(model_path):
            log.info(
                "INFO: Skipped extraction. Archive already extracted in directory: %s",
                model_path,
            )
            return model_path

        with zipfile.ZipFile(zip_path, "r") as zip_f:
            zip_f.extractall(model_path)

        # path to model
        return model_path

    def resolve_and_load(
        self, path_or_name_or_model_reference: Union[str, ModuleBase], **kwargs
    ):
        """Try our best to load a model, given a path or a name. Simply returns any loaded model
        passed in. This exists to ease the burden on workflow developers who need to accept
        individual blocks in their API, where users may have references to custom models or may only
        have the ability to give the name of a stock model.

        Args:
            path_or_name_or_model_reference (str, ModuleBase): Either a
                - Path to a model on disk
                - Name of a model that the catalog knows about
                - Loaded module (e.g. block or workflow)
            **kwargs: Any keyword arguments to pass along to ModelManager.load()
                      or ModelManager.download()
                e.g. parent_dir

        Returns:
            A loaded module

        Examples:
            >>> stock_syntax_model = manager.resolve_and_load('syntax_izumo_en_stock')
            >>> local_categories_model = manager.resolve_and_load('path/to/categories/model')
            >>> some_custom_model = manager.resolve_and_load(some_custom_model)
        """
        error.type_check(
            "<COR50266694E>",
            str,
            ModuleBase,
            path_or_name_or_model_reference=path_or_name_or_model_reference,
        )

        # If this is already a module, we're good to go
        if isinstance(path_or_name_or_model_reference, ModuleBase):
            log.debug("Returning model %s directly", path_or_name_or_model_reference)
            return path_or_name_or_model_reference

        # Otherwise, this could either be a path on disk or some name of a model that our catalog
        # can resolve and fetch
        if os.path.isdir(path_or_name_or_model_reference):
            # Try to load from path
            log.debug(
                "Attempting to load model from path %s", path_or_name_or_model_reference
            )
            return self.load(path_or_name_or_model_reference, **kwargs)

        error(
            "<COR50207495E>",
            ValueError(
                "could not find model with name `{}`".format(
                    path_or_name_or_model_reference
                )
            ),
        )

    def get_singleton_model_cache_info(self):
        """Returns information about the singleton cache in {hash: module type} format

        Returns:
            Dict[str, type]
                A dictionary of model hashes to model types
        """
        return {k: type(v) for k, v in self.singleton_module_cache.items()}

    def clear_singleton_cache(self):
        """Clears the cache of singleton models. Useful to release references of models, as long as
        you know that they are no longer held elsewhere and you won't be loading them again.

        Returns:
            None
        """
        self.singleton_module_cache = {}

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

    @classmethod
    def get_module_class_from_config(cls, module_config):
        """Utility function to fetch module class information
        from ModuleConfig

        Args:
            module_config: caikit.core.module.ModuleConfig
                Configuration for caikit.core module
        Returns:
            str: name of the module_class
        """
        module_class = ""
        for module_type in _MODULE_TYPES:
            module_class_name = "{}_class".format(module_type.lower())
            if module_class_name in module_config:
                module_class = module_config.get(module_class_name)
                break

        return module_class
