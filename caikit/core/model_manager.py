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
from typing import Dict, Optional, Type, Union
import os
import tempfile
import zipfile

# First Party
import alog

# Local
from ..interfaces.common.data_model.stream_sources import S3Path
from .model_management import (
    ModelFinderBase,
    ModelInitializerBase,
    ModelTrainerBase,
    model_finder_factory,
    model_initializer_factory,
    model_trainer_factory,
)
from .modules.base import ModuleBase
from .registries import module_registry
from .toolkit.errors import error_handler
from .toolkit.factory import Factory, FactoryConstructible
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
        self._singleton_module_cache = {}
        self._trainers = {}
        self._finders = {}
        self._initializers = {}
        self.__singleton_lock = Lock()

    def initialize_components(self):
        """Proactively initialize all configured trainer/finder/initializer
        component instances. This is a separate call to enable explicit config.
        """
        # Initialize all configured components
        mm_config = get_config().model_management
        for trainer in mm_config.trainers:
            self._get_trainer(trainer)
        for finder in mm_config.finders:
            self._get_finder(finder)
        for initializer in mm_config.initializers:
            self._get_initializer(initializer)

    ## Public ##################################################################

    def train(
        self,
        module: Union[Type[ModuleBase], str],
        *args,
        trainer: Union[str, ModelTrainerBase] = "default",
        save_path: Optional[Union[str, S3Path]] = None,
        save_with_id: bool = False,
        wait: bool = False,
        **kwargs,
    ) -> ModelTrainerBase.ModelFutureBase:
        """Train an instance of the given module with the given args and kwargs
        using the given trainer.

        Each module's train function encapsulates the code needed to perform the
        training locally. This top-level train function provides the wrapper
        functionality to delegate the execution of the module's train function
        to an alternate framework using a ModelTrainerBase. It also allows
        training to be launched asynchronously.

        Args:
            module (Union[Type[ModuleBase], str]): The module class or guid for
                the module to train
            *args: Additional positional args to pass through to the module's
                train function

        Kwargs:
            trainer (Union[str, ModelTrainerBase]): The trainer to use. If given
                as a string, this is a key in the global config at
                model_management.trainers.
            save_path (Optional[Union[str, S3Path]]): Path where the model should be
                saved (may be relative to a remote trainer's filesystem, or link to S3
                storage)
            save_with_id (bool): Inject the training ID into the save path for
                the output model
            wait (bool): Wait for training to complete before returning
            **kwargs: Additional keyword arguments to pass through to the
                modules's train function

        Returns:
            model_future (ModelTrainerBase.ModelFutureBase): The future handle
                to the model which holds the status of the in-flight training.
        """
        # Resolve the module class
        if isinstance(module, str):
            module_id = module
            module = module_registry().get(module_id)
            error.value_check(
                "<COR00469102E>",
                module is not None,
                "Unable to train unknown module {}",
                module_id,
            )
        error.subclass_check("<COR05418775E>", module, ModuleBase)

        # Get the trainer to use
        trainer: ModelTrainerBase = self._get_trainer(trainer)

        # Start the training
        with alog.ContextTimer(log.debug, "Started training in: "):
            model_future = trainer.train(
                module,
                *args,
                save_path=save_path,
                save_with_id=save_with_id,
                **kwargs,
            )
            log.debug(
                "Started training %s with save path %s",
                model_future.id,
                model_future.save_path,
            )

        # If requested, wait for the future to complete
        if wait:
            log.debug("Waiting for training %s to complete", model_future.id)
            with alog.ContextTimer(
                log.debug, "Finished training %s in: ", model_future.id
            ):
                model_future.wait()

        # Return a handle to the training
        return model_future

    def get_model_future(
        self,
        training_id: str,
    ) -> ModelTrainerBase.ModelFutureBase:
        """Get the future handle to an in-progress training

        Args:
            training_id (str): The ID string from the original training
                submission's ModelFuture

        Returns:
            model_future (ModelTrainerBase.ModelFutureBase): The future handle
                to the model which holds the status of the in-flight training.
        """
        try:
            trainer = self._get_trainer(ModelTrainerBase.get_trainer_name(training_id))

        # Fall back to the default trainer to try to find this ID
        except ValueError:
            trainer = self._get_trainer("default")

        return trainer.get_model_future(training_id)

    def load(
        self,
        module_path: str,
        *,
        load_singleton: bool = False,
        finder: Union[str, ModelFinderBase] = "default",
        initializer: Union[str, ModelInitializerBase] = "default",
        **kwargs,
    ):
        """Load a model and return an instantiated object on which we can run
        inference.

        Args:
            module_path (str | BytesIO | bytes): A module path (identifier) to
                one of the following:
                1. A directory containing a yaml config file in the top level.
                2. A zip archive containing either a yaml config file in the
                    top level when extracted, or a directory containing a yaml
                    config file in the top level.
                3. A BytesIO object corresponding to a zip archive containing
                    either a yaml config file in the top level when extracted,
                    or a directory containing a yaml config file in the top
                    level.
                4. A bytes object corresponding to a zip archive containing
                    either a yaml config file in the top level when extracted,
                    or a directory containing a yaml config file in the top
                    level.
                5. A string that is understood by the configured
                    finder/initializer

        Kwargs:
            load_singleton (bool): Load this model as a singleton
            finder (Union[str, ModelFinderBase]): Finder to use when loading
                this model. If passed as a string, this names the finder in the
                global config model_management.finders section.
            initializer (Union[str, ModelInitializerBase]): Loader to use when
                initializint this model. If passed as a string, this is the name
                of the initializer in the global
                config model_management.initializers section.

        Returns:
            model (ModuleBase) Model object that is loaded, configured, and
                ready for prediction.
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
            return self._load_from_zipfile(
                module_path, load_singleton, finder, initializer, **kwargs
            )
        try:
            return self._load_from_dir(
                module_path, load_singleton, finder, initializer, **kwargs
            )
        except FileNotFoundError:
            error(
                "<COR80419785E>",
                FileNotFoundError(
                    "Module load path `{}` does not contain a `config.yml` file.".format(
                        module_path
                    )
                ),
            )

    def extract(
        self, zip_path: str, model_path: str, force_overwrite: bool = False
    ) -> str:
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
        return {k: type(v) for k, v in self._singleton_module_cache.items()}

    def clear_singleton_cache(self):
        """Clears the cache of singleton models. Useful to release references of models, as long as
        you know that they are no longer held elsewhere and you won't be loading them again.

        Returns:
            None
        """
        with self.__singleton_lock:
            self._singleton_module_cache.clear()

    ## Implementation Details ##################################################

    def _load_from_dir(
        self, module_path, load_singleton, finder, initializer, **kwargs
    ):
        """Load a model from a directory.

        Args:
            module_path (str): Path to directory. At the top level of directory
                is `config.yml` which holds info about the model.
            load_singleton (bool): Load this model as a singleton
            finder (Union[str, ModelFinderBase]): Finder to use when loading
                this model. If passed as a string, this names the finder in the
                global config model_management.finders section.
            initializer (Union[str, ModelInitializerBase]): Loader to use when
                loading this model. If passed as a string, this is the name of
                the initializer in the global
                config model_management.initializers section.

        Returns:
            subclass of caikit.core.modules.ModuleBase: Model object that is
                loaded, configured, and ready for prediction.
        """
        with self._singleton_lock(load_singleton):
            if singleton_entry := (
                load_singleton and self._singleton_module_cache.get(module_path)
            ):
                log.debug("Found %s in the singleton cache", module_path)
                return singleton_entry

            # Use the given finder to try to find the module config for this
            # module_path
            #
            # NOTE: This will lazily construct named finders if needed
            log.debug("Attempting to find [%s] with finder %s", module_path, finder)
            finder = self._get_finder(finder)
            model_config = finder.find_model(module_path, **kwargs)
            error.value_check(
                "<COR92173495E>",
                model_config is not None,
                "Unable to find a ModuleConfig for {}",
                module_path,
            )

            # Use the given initializer to try to load the model
            #
            # NOTE: This will lazily construct named initializers if needed
            initializer = self._get_initializer(initializer)
            loaded_model = initializer.init(model_config, **kwargs)
            error.value_check(
                "<COR50207494E>",
                loaded_model is not None,
                "Unable to load model from {} with MODULE_ID {}",
                module_path,
                model_config.module_id,
            )

            # If loading as a singleton, populate the cache
            if load_singleton:
                self._singleton_module_cache[module_path] = loaded_model

            # Return successfully!
            return loaded_model

    def _load_from_zipfile(
        self, module_path, load_singleton, finder, initializer, **kwargs
    ):
        """Load a model from a zip archive.

        Args:
            module_path (str): Path to directory. At the top level of directory
                is `config.yml` which holds info about the model.
            load_singleton (bool): Load this model as a singleton
            finder (Union[str, ModelFinderBase]): Finder to use when loading
                this model. If passed as a string, this names the finder in the
                global config model_management.finders section.
            initializer (Union[str, ModelInitializerBase]): Loader to use when
                loading this model. If passed as a string, this is the name of
                the initializer in the global
                config model_management.initializers section.

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
                model = self._load_from_dir(
                    extract_path, load_singleton, finder, initializer, **kwargs
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
                        nested_dirs[0], load_singleton, finder, initializer, **kwargs
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

    @contextmanager
    def _singleton_lock(self, load_singleton: bool):
        """Helper contextmanager that will only lock the singleton cache if this
        load is a singleton load
        """
        if load_singleton:
            with self.__singleton_lock:
                yield
        else:
            yield

    @staticmethod
    def _get_component(
        component: Union[str, FactoryConstructible],
        component_dict: Dict[str, FactoryConstructible],
        component_factory: Factory,
        component_name: str,
        component_cfg: dict,
        component_type: type,
    ) -> FactoryConstructible:
        """Common logic for resolving components from config

        NOTE: This is done lazily to avoid relying on import order and to allow
            for dynamic config changes
        """
        error.type_check(
            "<COR45466249E>", str, component_type, **{component_name: component}
        )
        if isinstance(component, component_type):
            return component
        if component not in component_dict:
            cfg = component_cfg.get(component)
            error.value_check(
                "<COR55057389E>",
                isinstance(cfg, dict),
                "Unknown {}: {}",
                component_name,
                component,
            )
            component_dict[component] = component_factory.construct(cfg, component)
        return component_dict[component]

    def _get_trainer(self, trainer: Union[str, ModelTrainerBase]) -> ModelTrainerBase:
        """Get the configured model trainer or the one passed by value"""
        return self._get_component(
            component=trainer,
            component_dict=self._trainers,
            component_factory=model_trainer_factory,
            component_name="trainer",
            component_cfg=get_config().model_management.trainers,
            component_type=ModelTrainerBase,
        )

    def _get_finder(self, finder: Union[str, ModelFinderBase]) -> ModelFinderBase:
        """Get the configured model finder or the one passed by value"""
        return self._get_component(
            component=finder,
            component_dict=self._finders,
            component_factory=model_finder_factory,
            component_name="finder",
            component_cfg=get_config().model_management.finders,
            component_type=ModelFinderBase,
        )

    def _get_initializer(
        self, initializer: Union[str, ModelInitializerBase]
    ) -> ModelInitializerBase:
        """Get the configured model initializer or the one passed by value"""
        return self._get_component(
            component=initializer,
            component_dict=self._initializers,
            component_factory=model_initializer_factory,
            component_name="initializer",
            component_cfg=get_config().model_management.initializers,
            component_type=ModelInitializerBase,
        )
