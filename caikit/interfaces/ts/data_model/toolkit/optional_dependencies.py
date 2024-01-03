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
This module encapsulates the core optional dependencies using lazy imports.
"""

# Standard
from types import ModuleType
from typing import Optional
import importlib

# First Party
import alog

log = alog.use_channel("OPTDEP")


## Implementation ##############################################################


class LazyModule(ModuleType):
    """A LazyModule is a module subclass that wraps another module but imports
    it lazily and then aliases __getattr__ to the lazily imported module.
    """

    def __init__(self, name: str, package: Optional[str] = None):
        """Hang onto the import args to use lazily"""
        self.__name = name
        self.__package = package
        self.__wrapped_module = None

    def __getattr__(self, name: str) -> any:
        """When asked for an attribute, make sure the wrapped module is imported
        and then delegate
        """
        if self.__wrapped_module is None:
            log.debug1("Triggering lazy import for %s.%s", self.__package, self.__name)
            self.__wrapped_module = importlib.import_module(
                self.__name,
                self.__package,
            )
        return getattr(self.__wrapped_module, name)


def have_module(name: str, package: Optional[str] = None) -> bool:
    """This method can be used to check whether a given optional dependency is
    available and should primarily be used for assertions when coding
    defensively.

    NOTE: Nested modules WILL force the import of parent modules

    TODO: Move this to import_tracker

    Args:
        name:  str
            The name of the module
        package:  Optional[str]
            The qualifying package for the module under investigation

    Returns:
        have_module:  bool
            True if the module can be imported, False otherwise
    """
    spec = importlib.util.find_spec(name, package)
    return (
        # No spec found under standard import
        spec is not None
        and spec.loader is not None
        and
        # Spec not found and delegated to import_tracker lazy failures
        spec.loader.__module__.split(".")[0] != "import_tracker"
    )


## Public ######################################################################

# The core optional dependencies
pd = LazyModule("pandas")
pyspark = LazyModule("pyspark")

# Import-time checks for the presence of optional dependencies
HAVE_NUMPY = have_module("numpy")
HAVE_PANDAS = have_module("pandas")
HAVE_PYSPARK = have_module("pyspark")
