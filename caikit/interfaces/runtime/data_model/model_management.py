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
Data model objects for the model management service
"""
# Standard
from typing import List

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber

# Local
from ....core.data_model import DataObjectBase, dataobject
from ...common.data_model import File
from .package import RUNTIME_PACKAGE


@dataobject(RUNTIME_PACKAGE)
class DeployModelRequest(DataObjectBase):
    """Request to deploy a model"""

    model_id: Annotated[str, FieldNumber(1)]
    model_files: Annotated[List[File], FieldNumber(2)]


@dataobject(RUNTIME_PACKAGE)
class UndeployModelRequest(DataObjectBase):
    """Request to undeploy a model"""

    model_id: Annotated[str, FieldNumber(1)]
