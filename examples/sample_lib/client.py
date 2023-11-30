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
from pathlib import Path
from time import sleep
import json
import os
import sys

# Third Party
import grpc
import requests

# Local
from caikit.config.config import get_config
from caikit.core.data_model.base import DataBase
from caikit.interfaces.runtime.data_model import (
    TrainingInfoRequest,
    TrainingStatusResponse,
)
from caikit.runtime import get_inference_request, get_train_params, get_train_request
from caikit.runtime.service_factory import ServicePackageFactory
import caikit

if __name__ == "__main__":
    # pylint: disable=duplicate-code
    caikit.config.configure(
        config_dict={
            "merge_strategy": "merge",
            "runtime": {
                "library": "sample_lib",
                "grpc": {"enabled": True},
                "http": {"enabled": True},
                "service_generation": {
                    "package": "caikit_sample_lib"
                },  # Same as the server
            },
        }
    )
    # Make sample_lib available for import
    sys.path.append(
        os.path.join(Path(__file__).parent.parent.parent, "tests/fixtures"),
    )
    # Local
    # pylint: disable=import-error
    from sample_lib.data_model import (
        SampleInputType,  # pylint: disable=import-outside-toplevel
    )

    # pylint: disable=import-error
    from sample_lib.modules import (
        SampleModule,  # pylint: disable=import-outside-toplevel
    )

    model_id = "my_model"

    if get_config().runtime.grpc.enabled:
        inference_service = ServicePackageFactory().get_service_package(
            ServicePackageFactory.ServiceType.INFERENCE,
        )
        train_service = ServicePackageFactory().get_service_package(
            ServicePackageFactory.ServiceType.TRAINING,
        )
        train_mgt_svc = ServicePackageFactory.get_service_package(
            ServicePackageFactory.ServiceType.TRAINING_MANAGEMENT,
        )
        # Setup the client
        port = 8085
        channel = grpc.insecure_channel(f"localhost:{port}")
        training_stub = train_service.stub_class(channel)
        training_management_stub = train_mgt_svc.stub_class(channel)
        inference_stub = inference_service.stub_class(channel)

        # send train request
        stream_type = (
            caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
        )
        training_data = stream_type(
            file=stream_type.FileReference(filename="protos/sample.json")
        )
        train_class = DataBase.get_class_for_name("SampleTaskSampleModuleTrainRequest")
        train_request_params_class = DataBase.get_class_for_name(
            "SampleTaskSampleModuleTrainParameters"
        )
        request = get_train_request(SampleModule)(
            model_name=model_id,
            parameters=get_train_params(SampleModule)(training_data=training_data),
        )
        print(request)
        response = training_stub.SampleTaskSampleModuleTrain(request.to_proto())
        print("*" * 30)
        print("RESPONSE from TRAIN gRPC\n")
        print(response)
        print("*" * 30)

        # wait for model to auto-load, "lazy_load_local_models" is True on server
        sleep(1)

        # Check training status
        training_info_request = TrainingInfoRequest(training_id=response.training_id)
        training_management_response = TrainingStatusResponse.from_proto(
            training_management_stub.GetTrainingStatus(training_info_request.to_proto())
        )

        print("*" * 30)
        print("RESPONSE from TRAIN MANAGEMENT gRPC\n")
        print(training_management_response)
        print("*" * 30)

        sleep(1)

        # send inference request on trained model
        predict_class = get_inference_request(SampleModule)
        sample_input = SampleInputType(name="world")

        request = predict_class(sample_input=sample_input)
        response = inference_stub.SampleTaskPredict(
            request.to_proto(), metadata=[("mm-model-id", model_id)], timeout=1
        )
        print("*" * 30)
        print("RESPONSE from INFERENCE gRPC\n")
        print(response)
        print("*" * 30)

    if get_config().runtime.http.enabled:
        # TODO: add train REST calls here

        # For now this assumes you have the model trained using the gRPC steps above
        port = 8080
        # Run inference for sample text
        payload = {"inputs": {"name": "world"}, "model_id": model_id}
        response = requests.post(
            f"http://localhost:{port}/api/v1/task/sample",
            json=payload,
            timeout=1,
        )
        print("*" * 30)
        print("RESPONSE from INFERENCE HTTP\n")
        print(json.dumps(response.json(), indent=4))
        print("*" * 30)
