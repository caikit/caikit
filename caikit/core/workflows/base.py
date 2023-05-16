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


"""This contains the `base` class from which **all** Workflows inherit. This
class is not for direct use and most methods are, in fact, abstract and
inherited from ModuleBase.
"""

# Standard
import os

# First Party
import alog

# Local
from ... import core
from .. import module as mod
from caikit.core.modules.decorator import module_type
from ..toolkit.errors import error_handler

log = alog.use_channel("WFBASE")
error = error_handler.get(log)


@module_type("workflow")
class WorkflowBase(mod.ModuleBase):
    """Abstract base class for creating Workflows.  Inherits from ModuleBase."""

    @classmethod
    # pylint: disable=arguments-differ
    def load(cls, model_path, *args, **kwargs):
        """Load a new instance of workflow from a given model_path

        Args:
            model_path: str
                Path to workflow
        Returns:
            caikit.core.workflows.base.WorkflowBase
                A new instance of any given implementation of WorkflowBase
        """
        return cls._load(WorkflowLoader(cls, model_path), *args, **kwargs)


# Hoist the @workflow decorator
workflow = WorkflowBase.workflow


class WorkflowSaver(mod.ModuleSaver):
    """A workflow saver that inherits from the module saver. Workflows should have a workflow_id
    and can directly save other modules (e.g., blocks).
    """

    MODULE_PATHS_KEY = "module_paths"

    def __init__(self, module, **kwargs):
        """Construct a new workflow saver.

        Args:
            module:  caikit.core.module.Module
                The instance of the module to be saved.
        """
        super().__init__(module, **kwargs)
        if not hasattr(module, "WORKFLOW_ID"):
            msg = "module `{}` is not a workflow.".format(module.__class__.__name__)
            log.warning("<COR80155031W>", msg)

        # `module_paths` are reserved, strip from config if they exist
        # (not ideal that they made it here, could be better to fix that that upstream)
        self.config.pop(WorkflowSaver.MODULE_PATHS_KEY, None)

    def save_module(self, module, relative_path):
        """Save a CaikitCore module within a workflow artifact and add a reference to the config.

        Args:
            module:  caikit.core.ModuleBase
                The CaikitCore module to save as part of this workflow
            relative_path:  str
                The relative path inside of `model_path` where the block will be saved
        """

        if not issubclass(module.__class__, mod.ModuleBase):
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
        self.config.setdefault(WorkflowSaver.MODULE_PATHS_KEY, {}).update(
            {relative_path: rel_path}
        )
        return rel_path, abs_path

    def save_module_list(self, modules, config_key):
        """Save a list of CaikitCore modules within a workflow artifact and add a reference to the
        config.

        Args:
            modules:  dict{str -> caikit.core.ModuleBase}
                A dict with module relative path as key and a CaikitCore module as value to save as
                part of this workflow
            config_key:  str
                The config key inside of `model_path` where the modules' relative path with be
                referenced

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
            if not issubclass(module.__class__, mod.ModuleBase):
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
        self.config.setdefault(WorkflowSaver.MODULE_PATHS_KEY, {}).update(
            {config_key: list_of_rel_path}
        )
        return list_of_rel_path, list_of_abs_path

    def save_params(self, **kwargs):
        """Save parameters in a workflow config

        Args:
            **kwargs: dict
                key-value pair of parameters to save in config.yml
        """
        self.config.update(kwargs)


class WorkflowLoader(mod.ModuleLoader):
    """A workflow loader that inherits from the module loader. Workflow loader is used to
    load internal blocks/resources and use them to instantiate a new instance of the
    module. Workflows should have a workflow_id.
    """

    def __init__(self, module, model_path):
        """Construct a new workflow loader.

        Args:
            module:  caikit.core.ModuleBase
                The CaikitCore module to load as part of this workflow
            model_path:  str
                The path to the directory where the workflow is to be loaded from.
        """
        super().__init__(model_path)
        if not hasattr(module, "WORKFLOW_ID"):
            msg = "module `{}` is not a workflow.".format(module.__class__.__name__)
            log.warning("<COR88287900W>", msg)

    def load_module(self, module_paths_key, load_singleton=False):
        """Load a CaikitCore module from a workflow config.module_paths specification.

        Args:
            module_paths_key:  str
                key in `config.module_paths` looked at to load a block/resource
        """
        # Load module from a given relative path
        # Can be updated to load from a block/resource key
        if "module_paths" not in self.config:
            error(
                "<COR08580509E>", KeyError("Missing `module_paths` in workflow config!")
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
            module_paths_key:  str
                key in `config.module_paths` looked at to load a list of block/resource

        Returns:
            list
                list of loaded modules
        """
        # Load module from a given relative path
        # Can be updated to load from a block/resource key
        if WorkflowSaver.MODULE_PATHS_KEY not in self.config:
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
