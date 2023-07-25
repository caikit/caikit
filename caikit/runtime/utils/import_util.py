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
"""A generic module to help import dynamic modules
"""

# Standard
from types import ModuleType
from typing import Any, List
import importlib
import re
import sys

# Third Party
from grpc import StatusCode

# First Party
import aconfig
import alog

# Local
from caikit import get_config
from caikit.core import MODEL_MANAGER
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
import caikit.core

log = alog.use_channel("COM-LIB-INIT")


class UnifiedDataModel:
    """This class acts as an aggregator between the data models of multiple
    Caikit libraries.
    """

    def __init__(self):
        """Initialize the internal set of libraries"""
        self._libraries = {}

    def add_library(self, lib_name: str, lib: ModuleType):
        """Add the given library to the unified data model"""
        if lib_name in self._libraries:
            raise ValueError(f"Double registration of {lib_name}")
        self._libraries[lib_name] = getattr(lib, "data_model", None)

    def __getattr__(self, name: str) -> Any:
        """Fetch an attribute by name aliasing into child libraries"""
        if name in self._libraries:
            return self._libraries[name]
        candidates = {
            lib_name: getattr(lib, name)
            for lib_name, lib in self._libraries.items()
            if hasattr(lib, name)
        }
        if len(candidates) == 1:
            return list(candidates.values())[0]
        if len(candidates) > 1:
            raise AttributeError(
                f"Multiple library implementations of {name} found: {list(candidates.keys())}"
            )
        return super().__getattr__(name)


def get_data_model(config: aconfig.Config = None) -> UnifiedDataModel:
    """
    Get the data model from the Caikit library of interest. This is accomplished
    via dynamic import on the caikit_library's environment variable.

    NOTE: This function also has the side-effect of importing each of the
        caikit_library libraries for the first time, causing their modules to
        be registered with the caikit.core module registry. It is a critical
        step in initializing the set of modules that can be loaded by this
        running server instance.

    Args:
        config(aconfig.Config): caikit configuration

    Returns:
        (module): Handle to the module after dynamic wild import
    """
    config = config or get_config()
    lib_names = clean_lib_names(config.runtime.library)

    # Add all caikit.interfaces.X modules
    lib_names.extend(
        [
            lib_name
            for lib_name in sys.modules
            if lib_name.startswith("caikit.interfaces.") and lib_name.count(".") == 2
        ]
    )

    cdm = UnifiedDataModel()
    for lib_name in lib_names:
        cdm = _get_cdm_from_lib(lib_name, cdm)

    # Check module registry to get base modules
    # NOTE: Following is done after previous for-loop, since we need to import
    # all the libraries which will register all the modules and that happens
    # in get_dynamic_module above
    base_lib_names = set()
    for module_class in caikit.core.registries.module_registry().values():
        lib_name = module_class.__module__.partition(".")[0]
        if lib_name not in lib_names:
            # This module is from a library not mentioned
            # in lib_names. Consider this as a base library, like caikit.core
            base_lib_names.add(lib_name)

    # Get data model from lib_names
    for lib_name in base_lib_names:
        cdm = _get_cdm_from_lib(lib_name, cdm)

    # Ensure that all model management components have been initialized
    # TODO: This function has a _ton_ of side effects! We need to split it up
    #   and isolate these side effects in appropriately named functions.
    #   Specifically, this function is not only responsible for creating the
    #   data model, but it performs the dynamic import of the domain library and
    #   initializes the model management components.
    MODEL_MANAGER.initialize_components()

    return cdm


def _get_cdm_from_lib(lib_name: str, cdm: UnifiedDataModel):
    """Function to get caikit core CDM from library name

    Args:
        lib_name (str): Caikit core library name
        cdm (UnifiedDataModel): Caikit core CDM
    Returns:
        cdm: UnifiedDataModel
    """
    caikit_library = get_dynamic_module(lib_name)

    if caikit_library is None:
        message = "Unable to load data model from library: %s" % (lib_name)
        log.error("<RUN22291311E>", message)
        raise ValueError(message)
    cdm.add_library(lib_name, caikit_library)
    return cdm


def get_dynamic_module(module_name: str, module_dir: str = None) -> ModuleType:
    """
    Get the dynamic module of interest.

    Args:
        module_name(str): Name of the module to be dynamically imported
        (Optional) module_dir(str): Name of the directory from where the module is
                to be dynamically imported

    Returns:
        (module): Handle to the module after dynamic import
    """
    if module_dir:
        module_path = module_dir + "." + module_name
    else:
        module_path = module_name
    log.info("<RUN11997772I>", "Loading service module: %s", module_path)
    # Try to find the spec for the module that we're interested in.
    spec = importlib.util.find_spec(module_path)
    if not spec:
        message = "Unable to find spec for module: %s" % (module_path)
        # TODO: figure out the better way of doing this
        # https://github.com/caikit/caikit/pull/85#discussion_r1182890609
        log.warning("<RUN11991313W>", message)
        raise CaikitRuntimeException(StatusCode.NOT_FOUND, message)
    # Found spec - import the library
    if module_dir:
        return importlib.import_module(module_path)

    return importlib.import_module(module_path, "*")


def clean_lib_names(caikit_library: str) -> List[str]:
    def clean(lib):
        # Regex explanation:
        # - Capturing group ([a-zA-Z_-]+)
        #    To match 1 or more of alphabets (small or uppercase) and symbols (-) and (_)
        # - Match special character `[` (0 or 1) times
        # - Match any number of characters present in the list [a-zA-Z_\.,]
        #   NOTE: . is special character thus prefixed by \
        # - Match special character `]` (0 or 1) times
        # - Match on of following symbols [<>=\d\.\-a-z] (\d is for digits)
        regex = r"([a-zA-Z_-]+)\[?[a-zA-Z_\.,]*\]?[<>=\d\.\-a-z]*"
        lib_name = re.search(regex, lib).group(1)
        cleaned = lib_name.replace("-", "_")  # replace hyphens with underscores
        return cleaned

    lib_names = caikit_library.split()
    return [clean(lib) for lib in lib_names]
