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
from enum import Enum

# First Party
import alog

# Local
from caikit.core.data_model import DataObjectBase, dataobject
from caikit.core.toolkit.wip_decorator import Action, WipCategory, work_in_progress

log = alog.use_channel("MDLOPS")

RUNTIME_PACKAGE = "caikit_data_model.runtime"


@work_in_progress(action=Action.WARNING, category=WipCategory.BETA)
@dataobject(RUNTIME_PACKAGE)
class TrainingInfoRequest(DataObjectBase):
    training_id: str


@work_in_progress(action=Action.WARNING, category=WipCategory.BETA)
@dataobject(RUNTIME_PACKAGE)
class TrainingJob(DataObjectBase):
    training_id: str
    model_name: str


@work_in_progress(action=Action.WARNING, category=WipCategory.BETA)
@dataobject(RUNTIME_PACKAGE)
class ModelPointer(DataObjectBase):
    model_id: str


@work_in_progress(action=Action.WARNING, category=WipCategory.BETA)
@dataobject(RUNTIME_PACKAGE)
class TrainingStatus(Enum):
    NOT_STARTED = 0
    HALTED = 1
    FAILED = 2
    DOWNLOADING = 3
    PROCESSING = 4
    STORING = 5
    COMPLETED = 6


@work_in_progress(action=Action.WARNING, category=WipCategory.BETA)
@dataobject(RUNTIME_PACKAGE)
class TrainingInfoResponse(DataObjectBase):
    training_id: str
    status: TrainingStatus
    submission_timestamp: str
    completion_timestamp: str
    error_code: str
