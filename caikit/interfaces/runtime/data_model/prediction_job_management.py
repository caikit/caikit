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
from datetime import datetime
from typing import List

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber

# Local
from ....core.data_model import DataObjectBase, JobStatus, dataobject
from .package import RUNTIME_PACKAGE


@dataobject(RUNTIME_PACKAGE)
class PredictionJobInfoRequest(DataObjectBase):
    """DataModel to request information about a PredictionJob"""

    prediction_id: Annotated[str, FieldNumber(1)]


@dataobject(RUNTIME_PACKAGE)
class PredictionJob(DataObjectBase):
    """DataModel returned as a result of starting a PredictionJob"""

    prediction_id: Annotated[str, FieldNumber(1)]


@dataobject(RUNTIME_PACKAGE)
class PredictionJobStatusResponse(DataObjectBase):
    """DataModel representing the status of a PredictionJob"""

    prediction_id: Annotated[str, FieldNumber(1)]
    state: Annotated[JobStatus, FieldNumber(2)]
    submission_timestamp: Annotated[datetime, FieldNumber(3)]
    completion_timestamp: Annotated[datetime, FieldNumber(4)]
    reasons: Annotated[List[str], FieldNumber(5)]
