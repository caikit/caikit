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

"""Local saver implementation for saving modules to disk.
Contains recursive functions for loading modules saved inside modules.
"""

# Standard
from importlib import metadata
import datetime
import os
import shutil
import uuid

# First Party
import alog

# Local
from ..modules.config import ModuleConfig
from ..toolkit import ObjectSerializer
from ..toolkit.errors import error_handler
from ..toolkit.wip_decorator import TempDisableWIP
from .base import ModuleBase
from .loader import ModuleLoader
from caikit.config import get_config

log = alog.use_channel("MODULE_SAVE")
error = error_handler.get(log)


class ModuleSaver:
    """A module saver that provides common functionality used for saving modules and also a context
    manager that cleans up gracefully in case an error is encountered during the save process.
    """

    SAVED_KEY_NAME = "saved"
    CREATED_KEY_NAME = "created"
    TRACKING_KEY_NAME = "tracking_id"
    MODULE_VERSION_KEY_NAME = "version"
    MODULE_ID_KEY_NAME = "module_id"
    MODULE_CLASS_KEY_NAME = "module_class"

    def __init__(self, module: ModuleBase, model_path):
        """Construct a new module saver.

        Args:
            module (caikit.core.module.Module): The instance of the module to be
                saved.
            model_path (str): The absolute path to the directory where the model
                will be saved.  If this directory does not exist, it will be
                created.
        """
        self.model_path = os.path.normpath(model_path)

        # Get possibly nested caikit library path
        module_path = module.__module__
        lib_name_generator = (
            k
            for k, v in get_config().libraries.items()
            if module_path.startswith(v.module_path)
        )
        try:
            self.library_name = next(lib_name_generator)
        except StopIteration:
            # This assumes no nested module path by default
            self.library_name = module_path.split(".")[0]  # tests

        try:
            self.library_version = metadata.version(self.library_name)
        except metadata.PackageNotFoundError:
            log.debug("<COR25991305D>", "No library version found")
            if (
                self.library_name in get_config().libraries
                and "version" in get_config().libraries[self.library_name]
            ):
                self.library_version = get_config().libraries[self.library_name].version
            else:
                self.library_version = "0.0.0"

        self.config = {
            self.library_name + "_version": self.library_version,
            self.CREATED_KEY_NAME: str(datetime.datetime.now()),
            self.SAVED_KEY_NAME: str(datetime.datetime.now()),
            "name": module.MODULE_NAME,
            self.TRACKING_KEY_NAME: str(uuid.uuid4()),
            self.MODULE_ID_KEY_NAME: module.MODULE_ID,
            self.MODULE_CLASS_KEY_NAME: module.MODULE_CLASS,
            self.MODULE_VERSION_KEY_NAME: module.MODULE_VERSION,
        }

        # Temp disable wip for following invocation to not log warnings for downstream
        # usage of ModuleSaver
        with TempDisableWIP():
            # Get metadata back about this module and add it to the config
            stored_config = module.metadata
        # Sanitize some things off of the config:
        # Remove the old `saved` timestamp:
        stored_config.pop(self.SAVED_KEY_NAME, None)
        # Remove any reserved keys, these will be set by the `ModuleConfig` class
        for key in ModuleConfig.reserved_keys:
            if key in stored_config:
                stored_config.pop(key)

        self.config.update(stored_config)

    def add_dir(self, relative_path, base_relative_path=""):
        """Create a directory inside the `model_path` for this saver.

        Args:
            relative_path (str): A path relative to this saver's `model_path`
                denoting the directory to create.
            base_relative_path (str): A path, relative to this saver's
                `model_path`, in which `relative_path` will be created.

        Returns:
            str, str: A tuple containing both the `relative_path` and
                `absolute_path` to the directory created.

        Examples:
            >>> with ModelSaver('/path/to/model') as saver:
            >>>     rel_path, abs_path = saver.add_dir('word_embeddings', 'model_data')
            >>> print(rel_path)
            model_data/word_embeddings
            >>> print(abs_path)
            /path/to/model/model_data/word_embeddings
        """
        base_relative_path = os.path.normpath(base_relative_path)
        relative_path = os.path.normpath(relative_path)

        relative_path = os.path.join(base_relative_path, relative_path)
        absolute_path = os.path.join(self.model_path, relative_path)

        os.makedirs(absolute_path, exist_ok=True)

        return relative_path, absolute_path

    def copy_file(self, file_path, relative_path=""):
        """Copy an external file into a subdirectory of the `model_path` for this saver.

        Args:
            file_path (str): Absolute path to the external file to copy.
            relative_path (str): The relative path inside of `model_path` where
                the file will be copied to. If set to the empty string (default)
                then the file will be placed directly in the `model_path`
                directory.

        Returns:
            str, str: A tuple containing both the `relative_path` and
                `absolute_path` to the copied file.
        """
        file_path = os.path.normpath(file_path)

        if not os.path.isfile(file_path):
            error(
                "<COR80954473E>",
                FileNotFoundError(
                    "Attempted to add `{}` but is not a regular file.".format(file_path)
                ),
            )

        filename = os.path.basename(os.path.normpath(file_path))

        relative_path, absolute_path = self.add_dir(relative_path)

        relative_file_path = os.path.join(relative_path, filename)
        absolute_file_path = os.path.join(absolute_path, filename)

        shutil.copyfile(file_path, absolute_file_path)

        return relative_file_path, absolute_file_path

    def save_object(self, obj, filename, serializer, relative_path=""):
        """Save a Python object using the provided ObjectSerializer.

        Args:
            obj (any): The Python object to save
            filename (str): The filename to use for the saved object
            serializer (ObjectSerializer): An ObjectSerializer instance (e.g.,
                YAMLSerializer) that should be used to serialize the object
            relative_path (str): The relative path inside of `model_path` where
                the object will be saved
        """
        if not issubclass(serializer.__class__, ObjectSerializer):
            error(
                "<COR85655282E>",
                TypeError(
                    "`{}` does not extend `ObjectSerializer`".format(
                        serializer.__class__.__name__
                    )
                ),
            )

        relative_path, absolute_path = self.add_dir(relative_path)

        # Normalize any '././' structure that may come from relative paths
        relative_file_path = os.path.normpath(os.path.join(relative_path, filename))
        absolute_file_path = os.path.normpath(os.path.join(absolute_path, filename))

        serializer.serialize(obj, absolute_file_path)

        return relative_file_path, absolute_file_path

    def update_config(self, additional_config):
        """Add items to this saver's config dictionary.

        Args:
            additional_config (dict): A dictionary of config options to add the
                this saver's configuration.

        Notes:
            The behavior of this method matches `dict.update` and is equivalent to calling
            `saver.config.update`.  The `saver.config` dictionary may be accessed directly for
            more sophisticated manipulation of the configuration.
        """
        self.config.update(additional_config)

    def save_module(self, module, relative_path):
        """Save a CaikitCore module within a workflow artifact and add a reference to the config.

        Args:
            module (caikit.core.ModuleBase): The CaikitCore module to save as
                part of this workflow
            relative_path (str): The relative path inside of `model_path` where
                the module will be saved
        """

        if not issubclass(module.__class__, ModuleBase):
            error(
                "<COR30664151E>",
                TypeError(
                    "`{}` does not extend `ModuleBase`".format(
                        module.__class__.__name__
                    )
                ),
            )

        rel_path, abs_path = self.add_dir(relative_path)
        # Save this module at the specified location
        module.save(abs_path)
        self.config.setdefault(ModuleLoader.MODULE_PATHS_KEY, {}).update(
            {relative_path: rel_path}
        )
        return rel_path, abs_path

    def save_module_list(self, modules, config_key):
        """Save a list of CaikitCore modules within a workflow artifact and add a reference to the
        config.

        Args:
            modules (dict{str -> caikit.core.ModuleBase}): A dict with module
                relative path as key and a CaikitCore module as value to save as
                part of this workflow
            config_key (str): The config key inside of `model_path` where the
                modules' relative path with be referenced

        Returns:
            list_of_rel_path: list(str)
                List of relative paths where the modules are saved
            list_of_abs_path: list(str)
                List of absolute paths where the modules are saved
        """
        # validate type of input parameters
        error.type_check("<COR44644420E>", dict, modules=modules)
        error.type_check("<COR54316176E>", str, config_key=config_key)

        list_of_rel_path = []
        list_of_abs_path = []

        # iterate through the dict and serialize the modules in its corresponding paths
        for relative_path, module in modules.items():
            if not issubclass(module.__class__, ModuleBase):
                error(
                    "<COR67834055E>",
                    TypeError(
                        "`{}` does not extend `ModuleBase`".format(
                            module.__class__.__name__
                        )
                    ),
                )
            error.type_check("<COR48984754E>", str, relative_path=relative_path)

            rel_path, abs_path = self.add_dir(relative_path)

            # Save this module at the specified location
            module.save(abs_path)

            # append relative and absolute path to a list that will be returned
            list_of_rel_path.append(rel_path)
            list_of_abs_path.append(abs_path)

        # update the config with config key and list of relative path
        self.config.setdefault(ModuleLoader.MODULE_PATHS_KEY, {}).update(
            {config_key: list_of_rel_path}
        )
        return list_of_rel_path, list_of_abs_path

    def __enter__(self):
        """Enter the module saver context.  This creates the `model_path` directory.  If this
        context successfully exits, then the model configuration and all files it contains will
        be written and saved to disk inside the `model_path` directory.  If any uncaught exceptions
        are thrown inside this context, then `model_path` will be removed.
        """
        os.makedirs(self.model_path, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the module saver context. If this context successfully exits, then the model
        configuration and all files it contains will be written and saved to disk inside the
        `model_path` directory.  If any uncaught exceptions are thrown inside this context, then
        `model_path` will be removed.
        """
        if exc_type is not None:
            shutil.rmtree(self.model_path, ignore_errors=True)
            return

        ModuleConfig(self.config).save(self.model_path)
