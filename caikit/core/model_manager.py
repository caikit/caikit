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
from contextlib import contextmanager
from io import BytesIO
from threading import Lock
from typing import List, Union
import os
import tempfile
import zipfile

# First Party
import alog

# Local
from .model_management import (
    ModelFinderBase,
    ModelLoaderBase,
    model_finder_factory,
    model_loader_factory,
)
from .modules.base import ModuleBase
from .modules.decorator import SUPPORTED_LOAD_BACKENDS_VAR_NAME
from .registries import module_registry
from .toolkit.errors import error_handler
from caikit.config import get_config

log = alog.use_channel("MDLMNG")
error = error_handler.get(log)

# restrict functions that are imported so we don't pollute the base module namespace
__all__ = [
    "get_valid_module_ids",
    "ModelManager",
]


def get_valid_module_ids():
    """Get a dictionary mapping all module IDs to the string names of the
    implementing classes.
    """
    return {
        module_id: model_class.__name__
        for module_id, model_class in module_registry().items()
    }


class ModelManager:
    """Manage the models or resources for library."""

    def __init__(self):
        """Initialize ModelManager."""
        # Map to store module caches, to be used for singleton model lookups
        self.singleton_module_cache = {}
        self._singleton_lock = Lock()
        self._finders = None
        self._loaders = None

    # make load function available from top-level of library
    def load(self, module_path, *, load_singleton=False, **kwargs):
        """Load a model and return an instantiated object on which we can run inference.

        Args:
            module_path (str | BytesIO | bytes): A module path to one of the
                following. 1. A module path to a directory containing a yaml
                config file in the top level. 2. A module path to a zip archive
                containing either a yaml config file in the top level when
                extracted, or a directory containing a yaml config file in the
                top level. 3. A BytesIO object corresponding to a zip archive
                containing either a yaml config file in the top level when
                extracted, or a directory containing a yaml config file in the
                top level. 4. A bytes object corresponding to a zip archive
                containing either a yaml config file in the top level when
                extracted, or a directory containing a yaml config file in the
                top level.
            load_singleton: bool (Defaults to False)
                Indicates whether this model should be loaded as a singleton.

        Returns:
            subclass of caikit.core.modules.ModuleBase: Model object that is
                loaded, configured, and ready for prediction.
        """
        error.type_check("<COR98255724E>", bool, load_singleton=load_singleton)

        # This allows a user to load their own model (e.g. model saved to disk)
        load_path = get_config().load_path
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
            return self._load_from_zipfile(module_path, load_singleton, **kwargs)
        try:
            return self._load_from_dir(module_path, load_singleton, **kwargs)
        except FileNotFoundError:
            error(
                "<COR80419785E>",
                FileNotFoundError(
                    "Module load path `{}` does not contain a `config.yml` file.".format(
                        module_path
                    )
                ),
            )

    def _load_from_dir(self, module_path, load_singleton, **kwargs):
        """Load a model from a directory.

        Args:
            module_path (str): Path to directory. At the top level of directory
                is `config.yml` which holds info about the model.
            load_singleton (bool): Indicates whether this model should be loaded
                as a singleton.

        Returns:
            subclass of caikit.core.modules.ModuleBase: Model object that is
                loaded, configured, and ready for prediction.
        """
        with self.singleton_lock(load_singleton):
            if singleton_entry := (
                load_singleton and self.singleton_module_cache.get(module_path)
            ):
                log.debug("Found %s in the singleton cache", module_path)
                return singleton_entry

            # Iterate all configured finders and try to find the model's config
            model_config = None
            finder_exceptions = []
            for finder in self._get_finders():
                model_config = finder.find_model(module_path, **kwargs)
                if isinstance(model_config, Exception):
                    finder_exceptions.append(model_config)
                    model_config = None
                elif model_config is not None:
                    log.debug(
                        "Successfully found %s with finder %s", module_path, finder.name
                    )
                    break

            # If model_config is None and there are exceptions, aggregate
            if model_config is None and finder_exceptions:
                self._raise_aggregate_error(
                    f"Failed to find {module_path}", finder_exceptions
                )
            error.value_check(
                "<COR92173495E>",
                model_config is not None,
                "Unable to find a ModuleConfig for {}",
                module_path,
            )

            # Iterate all loaders and try to load the model
            loaded_model = None
            loader_exceptions = []
            for loader in self._get_loaders():
                loaded_model = loader.load(model_config, **kwargs)
                if isinstance(loaded_model, Exception):
                    loader_exceptions.append(loaded_model)
                    loaded_model = None
                if loaded_model is not None:
                    log.debug(
                        "Successfully loaded %s with loader %s",
                        module_path,
                        loader.name,
                    )
                    break

            # If no model successfully loaded, it's an error
            if loaded_model is None and loader_exceptions:
                self._raise_aggregate_error(
                    f"Failed to load {module_path}", loader_exceptions
                )
            if loaded_model is None:
                error(
                    "<COR50207494E>",
                    ValueError(
                        f"Unable to load model from {module_path} with MODULE_ID {model_config.module_id}"
                    ),
                )

            # If loading as a singleton, populate the cache
            if load_singleton:
                self.singleton_module_cache[module_path] = loaded_model

            # Return successfully!
            return loaded_model

    def _load_from_zipfile(self, module_path, load_singleton, **kwargs):
        """Load a model from a zip archive.

        Args:
            module_path (str): Path to directory. At the top level of directory
                is `config.yml` which holds info about the model.
            load_singleton (bool): Indicates whether this model should be loaded
                as a singleton.

        Returns:
            subclass of caikit.core.modules.ModuleBase: Model object that is
                loaded, configured, and ready for prediction.
        """
        with tempfile.TemporaryDirectory() as extract_path:
            with zipfile.ZipFile(module_path, "r") as zip_f:
                zip_f.extractall(extract_path)
            # Depending on the way the zip archive is packaged, out temp directory may unpack
            # to files directly, or it may unpack to a (single) directory containing the files.
            # We expect the former, but fall back to the second if we can't find the config.
            try:
                model = self._load_from_dir(extract_path, load_singleton, **kwargs)
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
                        nested_dirs[0], load_singleton, **kwargs
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
            zip_path (str): Location of .zip file to extract.
            model_path (str): Model directory where the archive should be
                unzipped unzipped.
            force_overwrite: bool (Defaults to false)
                Force an overwrite to model_path, even if the folder exists
        Returns:
            str: Output path where the model archive is unzipped.
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
        individual modules in their API, where users may have references to custom models or may
        only have the ability to give the name of a stock model.

        Args:
            path_or_name_or_model_reference (str, ModuleBase): Either a
                - Path to a model on disk
                - Name of a model that the catalog knows about
                - Loaded module
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
            Dict[str, type]: A dictionary of model hashes to model types
        """
        return {k: type(v) for k, v in self.singleton_module_cache.items()}

    def clear_singleton_cache(self):
        """Clears the cache of singleton models. Useful to release references of models, as long as
        you know that they are no longer held elsewhere and you won't be loading them again.

        Returns:
            None
        """
        with self._singleton_lock:
            self.singleton_module_cache.clear()

    @contextmanager
    def singleton_lock(self, load_singleton: bool):
        """Helper contextmanager that will only lock the singleton cache if this
        load is a singleton load
        """
        if load_singleton:
            with self._singleton_lock:
                yield
        else:
            yield

    def _get_supported_load_backends(self, backend_impl: ModuleBase):
        """Function to get a list of supported load backends
        that the module supports

        Args:
            backend_impl (caikit.core.ModuleBase): Module implementing the
                backend
        Returns:
            list(backend_types): list of backends that are supported for model
                load
        """

        # Get list of backends that are supported for load
        # NOTE: since code in a module can change anytime, its support
        # for various backend might also change, in which case,
        # it would be better to keep the backend information in the model itself
        # If module_backend is None, then we will assume that this model is not loadable in
        # any other backend
        return getattr(backend_impl, SUPPORTED_LOAD_BACKENDS_VAR_NAME, [])

    def _get_finders(self) -> List[ModelFinderBase]:
        """Get the configured model finders

        NOTE: This is done lazily to avoid relying on import order and to allow
            for dynamic config changes
        """
        if self._finders is None:
            self._finders = [
                model_finder_factory.construct(finder_cfg)
                for finder_cfg in get_config().model_management.loading.finders
            ]
        return self._finders

    def _get_loaders(self) -> List[ModelLoaderBase]:
        """Get the configured model loaders

        NOTE: This is done lazily to avoid relying on import order and to allow
            for dynamic config changes
        """
        if self._loaders is None:
            self._loaders = [
                model_loader_factory.construct(loader_cfg)
                for loader_cfg in get_config().model_management.loading.loaders
            ]
        return self._loaders

    @staticmethod
    def _raise_aggregate_error(err_msg: str, exceptions: List[Exception]):
        """Common semantics for aggregating multiple errors"""
        error_types = set(type(err) for err in exceptions)
        error_type = ValueError
        if len(error_types) == 1:
            error_type = list(error_types)[0]
        if len(exceptions) == 1:
            raise error_type(err_msg) from exceptions[0]
        err = error_type(err_msg)
        err.parents = exceptions
        raise err
