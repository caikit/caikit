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

# First Party
import alog

# Local
from caikit.core.toolkit.wip_decorator import Action, WipCategory, work_in_progress
import caikit.core

log = alog.use_channel("MDLOPS")


@work_in_progress(action=Action.WARNING, category=WipCategory.BETA)
@caikit.core.dataobject(
    schema={"training_id": str},
    package="caikit_data_model.runtime",
)
class TrainingInfoRequest:
    pass


@work_in_progress(action=Action.WARNING, category=WipCategory.BETA)
@caikit.core.dataobject(
    schema={"training_id": str, "model_name": str},
    package="caikit_data_model.runtime",
)
class TrainingJob:
    pass


@work_in_progress(action=Action.WARNING, category=WipCategory.BETA)
@caikit.core.dataobject(
    schema={"model_id": str},
    package="caikit_data_model.runtime",
)
class ModelPointer:
    pass


@work_in_progress(action=Action.WARNING, category=WipCategory.BETA)
@caikit.core.dataobject(
    {
        "enum": [
            "NOT_STARTED",
            "HALTED",
            "FAILED",
            "DOWNLOADING",
            "PROCESSING",
            "STORING",
            "COMPLETED",
        ]
    }
)
class TrainingStatus:
    pass


@work_in_progress(action=Action.WARNING, category=WipCategory.BETA)
@caikit.core.dataobject(
    schema={
        "training_id": str,
        "status": TrainingStatus,
        "submission_timestamp": str,
        "completion_timestamp": str,
        "error_code": str,
    },
    package="caikit_data_model.runtime",
)
class TrainingInfoResponse:
    pass
