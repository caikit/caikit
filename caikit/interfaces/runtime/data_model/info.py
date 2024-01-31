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
from dataclasses import dataclass
from typing import Dict, List, Optional

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from caikit.core.data_model import PACKAGE_COMMON, DataObjectBase, dataobject

log = alog.use_channel("RUNTIMEOPS")

RUNTIME_PACKAGE = f"{PACKAGE_COMMON}.runtime"


@dataobject(RUNTIME_PACKAGE)
@dataclass
class RuntimeInfoRequest(DataObjectBase):
    """Empty request for runtime server information"""


@dataobject(RUNTIME_PACKAGE)
@dataclass
class RuntimeInfoResponse(DataObjectBase):
    runtime_version: Annotated[Optional[str], FieldNumber(1)]
    python_packages: Annotated[Dict[str, str], FieldNumber(2)]


@dataobject(RUNTIME_PACKAGE)
@dataclass
class ModelInfoRequest(DataObjectBase):
    """Empty request for runtime server information"""

    model_ids: Annotated[Optional[List[str]], FieldNumber(1)]


@dataobject(RUNTIME_PACKAGE)
@dataclass
class ModelInfo(DataObjectBase):
    """Information regarding a specific Model instance"""

    # Model information
    model_path: Annotated[str, FieldNumber(1)]
    name: Annotated[str, FieldNumber(2)]
    size: Annotated[int, FieldNumber(3)]
    metadata: Annotated[Dict[str, str], FieldNumber(4)]
    loaded: Annotated[bool, FieldNumber(7)]

    # Module Information
    module_id: Annotated[str, FieldNumber(5)]
    module_metadata: Annotated[Dict[str, str], FieldNumber(6)]


@dataobject(RUNTIME_PACKAGE)
@dataclass
class ModelInfoResponse(DataObjectBase):
    """Model Info response contains a list of ModelInfos"""

    models: Annotated[List[ModelInfo], FieldNumber(1)]
