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
from typing import Dict, Optional

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from caikit.core.data_model import PACKAGE_COMMON, DataObjectBase, dataobject

log = alog.use_channel("RUNTIMEOPS")

RUNTIME_PACKAGE = f"{PACKAGE_COMMON}.runtime"


@dataobject(RUNTIME_PACKAGE)
class RuntimeInfoRequest(DataObjectBase):
    """Empty request for runtime server information"""


@dataobject(RUNTIME_PACKAGE)
class RuntimeInfoResponse(DataObjectBase):
    runtime_version: Annotated[Optional[str], FieldNumber(1)]
    python_packages: Annotated[Dict[str, str], FieldNumber(2)]
