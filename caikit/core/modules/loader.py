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

"""Local loader implementation for loading modules from disk.
Contains recursive functions for loading modules saved inside modules.
"""

# Standard
import os

# First Party
import alog

# Local
from ..toolkit.errors import error_handler
from .config import ModuleConfig
from caikit import core

log = alog.use_channel("MODULE_LOAD")
error = error_handler.get(log)


class ModuleLoader:
    MODULE_PATHS_KEY = "module_paths"

    def __init__(self, model_path):
        """Construct a new module loader.

        Args:
            model_path (str): The path to the directory where the model is to be
                loaded from.
        """
        self.model_path = os.path.normpath(model_path)
        error.dir_check("<COR43014802E>", model_path)
        self.config = ModuleConfig.load(model_path)

    def load_arg(self, arg):
        """Extract arg value from the loaded model's config"""
        return getattr(self.config, arg)

    def load_args(self, *args):
        """Extract values from the loaded model's config"""
        return tuple(getattr(self.config, arg) for arg in args)

    def load_module(self, module_paths_key, load_singleton=False):
        """Load a CaikitCore module from a module config.module_paths specification.

        Args:
            module_paths_key (str): key in `config.module_paths` looked at to
                load a module
            load_singleton (bool): singleton load flag to pass to individual
                module loads
        """
        # Load module from a given relative path
        if "module_paths" not in self.config:
            error(
                "<COR08580509E>", KeyError("Missing `module_paths` in module config!")
            )

        if module_paths_key not in self.config.module_paths:
            error(
                "<COR22069088E>",
                KeyError(
                    "Missing required {} key in config.module_paths!".format(
                        module_paths_key
                    )
                ),
            )

        module_path = os.path.join(
            self.model_path, self.config.module_paths[module_paths_key]
        )
        return core.load(module_path, load_singleton=load_singleton)

    def load_module_list(self, module_paths_key):
        """Load a list of CaikitCore module from a workflow config.module_paths specification.

        Args:
            module_paths_key (str): key in `config.module_paths` looked at to
                load a list of modules

        Returns:
            list: list of loaded modules
        """
        # Load module from a given relative path
        # Can be updated to load from a module key
        if self.MODULE_PATHS_KEY not in self.config:
            error(
                "<COR52619266E>", KeyError("Missing `module_paths` in workflow config!")
            )

        if module_paths_key not in self.config.module_paths:
            error(
                "<COR75976687E>",
                KeyError(
                    "Missing required {} key in config.module_paths!".format(
                        module_paths_key
                    )
                ),
            )

        module_list = self.config.module_paths[module_paths_key]
        error.type_check("<COR21790391E>", list, module_list=module_list)

        # Iterate through the list and load module one by one
        loaded_modules = []
        for module in module_list:
            module_path = os.path.join(self.model_path, module)
            loaded_modules.append(core.load(module_path))

        return loaded_modules
