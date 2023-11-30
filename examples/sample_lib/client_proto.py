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

# This file demonstrates how to use compiled pb2s along with the DataModel for a library
# Standard
from pathlib import Path
from time import sleep
import os
import sys

# Third Party
import grpc

# Local
# pylint: disable=no-name-in-module,import-error
from .generated import samplelibservice_pb2_grpc, samplelibtrainingservice_pb2_grpc

# pylint: disable=no-name-in-module,import-error
from .generated.caikit_sample_lib import (
    sampletaskrequest_pb2,
    sampletasksamplemoduletrainparameters_pb2,
    sampletasksamplemoduletrainrequest_pb2,
)

# Make sample_lib available for import
sys.path.append(
    os.path.join(Path(__file__).parent.parent.parent, "tests/fixtures"),
)

# Local
# pylint: disable=wrong-import-position,wrong-import-order,import-error
import sample_lib.data_model as dm

if __name__ == "__main__":
    model_id = "my_model"

    # Setup the client
    port = 8085
    channel = grpc.insecure_channel(f"localhost:{port}")

    # send train request
    request = sampletasksamplemoduletrainrequest_pb2.SampleTaskSampleModuleTrainRequest(
        model_name=model_id,
        parameters=sampletasksamplemoduletrainparameters_pb2.SampleTaskSampleModuleTrainParameters(
            training_data={"file": {"filename": "protos/sample.json"}}
        ),
    )
    training_stub = samplelibtrainingservice_pb2_grpc.SampleLibTrainingServiceStub(
        channel=channel
    )
    response = training_stub.SampleTaskSampleModuleTrain(request)
    print("*" * 30)
    print("RESPONSE from TRAIN gRPC\n")
    print(response)
    print("*" * 30)

    sleep(1)

    sample_input = dm.SampleInputType(name="world")

    request = sampletaskrequest_pb2.SampleTaskRequest(
        sample_input=sample_input.to_proto()
    )
    inference_stub = samplelibservice_pb2_grpc.SampleLibServiceStub(channel=channel)
    response = inference_stub.SampleTaskPredict(
        request, metadata=[("mm-model-id", model_id)], timeout=1
    )
    print("*" * 30)
    print("RESPONSE from INFERENCE gRPC\n")
    print(response)
    print("*" * 30)
