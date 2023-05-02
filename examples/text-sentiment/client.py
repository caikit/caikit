import grpc
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


