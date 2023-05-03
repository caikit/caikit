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

# Third Party
import grpc

# Local
from caikit.runtime.service_factory import ServicePackageFactory
from text_sentiment.data_model import TextInput

inference_service = ServicePackageFactory().get_service_package(
    ServicePackageFactory.ServiceType.INFERENCE,
    ServicePackageFactory.ServiceSource.GENERATED,
)

port = 8085
channel = grpc.insecure_channel(f"localhost:{port}")

client_stub = inference_service.stub_class(channel)

# print(dir(client_stub))

for text in ["I am not feeling well today!", "Today is a nice sunny day"]:
    input_text_proto = TextInput(text=text).to_proto()
    request = inference_service.messages.HfBlockRequest(text_input=input_text_proto)
    response = client_stub.HfBlockPredict(
        request, metadata=[("mm-model-id", "text_sentiment")]
    )
    print("Text:", text)
    print("RESPONSE:", response)
