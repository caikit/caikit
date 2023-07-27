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

# Third Party
import grpc

# First Party
import alog

# Local
from caikit.core import MODEL_MANAGER
from caikit.interfaces.runtime.data_model import (
    TrainingInfoRequest,
    TrainingStatusResponse,
)
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException

log = alog.use_channel("MR-SERVICR-I")


class TrainingManagementServicerImpl:
    """This class contains the implementation of all of the RPCs that are required to run a
    service in Model Mesh as a Model-Runtime."""

    def GetTrainingStatus(self, request, context):  # pylint: disable=unused-argument
        """Missing associated documentation comment in .proto file."""
        training_info = TrainingInfoRequest.from_proto(request)

        try:
            model_future = MODEL_MANAGER.get_model_future(
                training_id=training_info.training_id
            )

            reasons = []
            if model_future.get_info().errors:
                reasons = [str(error) for error in model_future.get_info().errors]

            return TrainingStatusResponse(
                training_id=training_info.training_id,
                state=model_future.get_info().status,
                reasons=reasons,
            ).to_proto()
        except ValueError as err:
            raise CaikitRuntimeException(
                grpc.StatusCode.NOT_FOUND,
                "{} not found in the list of currently running training jobs".format(
                    training_info.training_id,
                ),
            ) from err
