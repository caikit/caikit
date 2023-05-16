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


"""This contains the `base` class from which ALL resource types inherit. This class is not for
direct use and most methods are, in fact, abstract.
"""

# First Party
import alog

# Local
from .. import module as mod
from caikit.core.modules.decorator import module_type
from ..toolkit.errors import error_handler

log = alog.use_channel("RSRCBASE")
error = error_handler.get(log)


@module_type("resource")
class ResourceBase(mod.ModuleBase):
    """Abstract base class for creating Resource Types.  Inherits from ModuleBase."""


# Hoist the @resource decorator
resource = ResourceBase.resource


class ResourceSaver(mod.ModuleSaver):
    """DEPRECATED. Use ModuleSaver directly"""
