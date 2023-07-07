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
from concurrent import futures

# Third Party
import grpc

# Local
from caikit.runtime.interceptors.caikit_runtime_server_wrapper import (
    CaikitRuntimeServerWrapper,
)
from caikit.runtime.service_generation.rpcs import TaskPredictRPC
from sample_lib.data_model import SampleOutputType


def test_rpc_is_passed_to_predict_handlers(sample_inference_service, open_port):
    calls = []

    def predict(request, context, caikit_rpc):
        calls.append(caikit_rpc)
        return SampleOutputType().to_proto()

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
    )
    wrapper = CaikitRuntimeServerWrapper(server, predict, sample_inference_service)
    sample_inference_service.registration_function(
        sample_inference_service.service, wrapper
    )
    wrapper.add_insecure_port(f"[::]:{open_port}")

    try:
        wrapper.start()

        client = sample_inference_service.stub_class(
            grpc.insecure_channel(f"localhost:{open_port}")
        )
        _ = client.SampleTaskPredict(
            sample_inference_service.messages.SampleTaskRequest(), timeout=3
        )
        assert len(calls) == 1
        assert isinstance(calls[0], TaskPredictRPC)
        assert calls[0].name == "SampleTaskPredict"
    finally:
        wrapper.stop(0)
