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

# Make generated available for import. This is needed because transitive
# dependencies are imported without any qualification in generated protobufs.
sys.path.extend(
    [
        os.path.join(Path(__file__).parent, "generated"),
    ]
)

# Local
from .generated import (
    caikit_data_model_sample_lib_pb2,
    caikit_sample_lib_pb2,
    caikit_sample_lib_pb2_grpc,
)

if __name__ == "__main__":
    model_id = "my_model"

    # Setup the client
    port = 8085
    channel = grpc.insecure_channel(f"localhost:{port}")

    # send train request
    request = caikit_sample_lib_pb2.SampleTaskSampleModuleTrainRequest(
        model_name=model_id,
        parameters=caikit_sample_lib_pb2.SampleTaskSampleModuleTrainParameters(
            training_data={"file": {"filename": "protos/sample.json"}}
        ),
    )
    training_stub = caikit_sample_lib_pb2_grpc.SampleLibTrainingServiceStub(
        channel=channel
    )
    response = training_stub.SampleTaskSampleModuleTrain(request)
    print("*" * 30)
    print("RESPONSE from TRAIN gRPC\n")
    print(response)
    print("*" * 30)

    sleep(1)

    sample_input = caikit_data_model_sample_lib_pb2.SampleInputType(name="world")

    request = caikit_sample_lib_pb2.SampleTaskRequest(sample_input=sample_input)
    inference_stub = caikit_sample_lib_pb2_grpc.SampleLibServiceStub(channel=channel)
    response = inference_stub.SampleTaskPredict(
        request, metadata=[("mm-model-id", model_id)], timeout=1
    )
    print("*" * 30)
    print("RESPONSE from INFERENCE gRPC\n")
    print(response)
    print("*" * 30)
