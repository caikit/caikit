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

# Standard
from dataclasses import dataclass
from typing import Optional, Type

# First Party
import alog

# Local
from caikit.core import ModuleBase

log = alog.use_channel("COREMODHELP")


@dataclass
class ModuleInfo:
    library: str
    kind: str
    type: str


def get_module_info(ck_module: Type[ModuleBase]) -> Optional[ModuleInfo]:
    """Determine the name for the module type for this Caikit Core Module. This
    is defined as the name string of the parent python module above the
    immediate parent. This is done so that Modules can end up in the same `module type`
    if they implement the same logical problem.

    The logic here assumes one of several conventions is followed for the Module
    1. The Module is declared in a python module named
        `<library>.modules.<module type>`
    2. The module derives from a base class that itself derives from one of the
        known type-hierarchy derived from `ModuleBase`.
    """
    # NOTE: all of this assumes <library> has no .
    # Use the library name to qualify the module type in case there are
    # collisions across domains (e.g. classification in nlp and cv)
    py_mod_name_parts = ck_module.__module__.split(".")
    lib_name = py_mod_name_parts[0]

    # First, look for the python module naming convention
    if len(py_mod_name_parts) >= 3:
        module_kind = py_mod_name_parts[1]
        module_type = py_mod_name_parts[2]
        log.debug3(
            "Using py-module module type %s and module kind %s for Module %s",
            module_type,
            module_kind,
            ck_module,
        )
        return ModuleInfo(library=lib_name, kind=module_kind, type=module_type)

    # Look for a base class that meets our expectations
    for parent in ck_module.__mro__[1:]:
        if not (
            parent.__module__.partition(".")[0] == "caikit"
            and parent.__module__.partition(".")[1] == "core"
        ):
            module_parts = parent.__module__.split(".")
            module_kind = module_parts[1]
            module_type = module_parts[-1]
            log.debug3(
                "Using parent module type %s and module kind %s for Module %s",
                module_type,
                module_kind,
                ck_module,
            )
            return ModuleInfo(library=lib_name, kind=module_kind, type=module_type)

    # We're out of luck!
    log.warning("Could not determine module type for %s", ck_module)
    return None
