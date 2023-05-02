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

# Have pylint ignore Class XXXX has no YYYY member so that we can use gRPC enums.
# pylint: disable=E1101

# First Party
import alog

# Local
from caikit.interfaces.runtime.data_model import (
    TrainingInfoRequest,
    TrainingInfoResponse,
)
from caikit.runtime.model_management.training_manager import TrainingManager

log = alog.use_channel("MR-SERVICR-I")


class TrainingManagementServicerImpl:
    """This class contains the implementation of all of the RPCs that are required to run a
    service in Model Mesh as a Model-Runtime."""

    def __init__(self):
        self.training_manager = TrainingManager.get_instance()

    def GetTrainingStatus(self, request, context):  # pylint: disable=unused-argument
        """Missing associated documentation comment in .proto file."""
        training_info = TrainingInfoRequest.from_proto(request)

        return TrainingInfoResponse(
            training_id=training_info.training_id,
            status=self.training_manager.get_training_status(
                training_info.training_id
            ).value,
        ).to_proto()
