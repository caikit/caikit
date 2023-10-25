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
This file contains interfaces to handle information requests
"""

# Standard
from typing import Dict

# First Party
import alog

# Local
from caikit.core.data_model import DataObjectBase, dataobject
from caikit.core.toolkit.wip_decorator import Action, WipCategory, work_in_progress

log = alog.use_channel("RUNTIMEOPS")

RUNTIME_PACKAGE = "caikit_data_model.runtime"


@work_in_progress(action=Action.WARNING, category=WipCategory.BETA)
@dataobject(RUNTIME_PACKAGE)
class RuntimeInfoRequest(DataObjectBase):
    pass


# TODO: what version details are in here -- is this meant to be more generic?
@work_in_progress(action=Action.WARNING, category=WipCategory.BETA)
@dataobject(RUNTIME_PACKAGE)
class RuntimeInfoResponse(DataObjectBase):
    version_info: Dict[str, str]
