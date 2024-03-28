**Table of contents**√
- [Interacting with the Sample lib](#interacting-with-the-sample-lib)
  - [Build and start a runtime server with sample\_lib](#build-and-start-a-runtime-server-with-sample_lib)
  - [Interact using the python client](#interact-using-the-python-client)
  - [Interact using terminal](#interact-using-terminal)
    - [To train a model](#to-train-a-model)
      - [Using gRPC](#using-grpc)
      - [Using HTTP](#using-http)
    - [To check on training status for a training](#to-check-on-training-status-for-a-training)
      - [Using gRPC](#using-grpc-1)
      - [Using HTTP](#using-http-1)
    - [To call inference on a model with model Id](#to-call-inference-on-a-model-with-model-id)
      - [To use the gRPC Server for inference](#to-use-the-grpc-server-for-inference)
      - [To use the REST Server for inference](#to-use-the-rest-server-for-inference)
  - [Interact using a combination of pb2s and DataModels](#interact-using-a-combination-of-pb2s-and-datamodels)


# Interacting with the Sample lib

This document describes how to quickly get a runtime server built with `sample_lib` library, train a model with gRPC and with that trained model, send an inference call to the server with either HTTP or gRPC call.

## Build and start a runtime server with sample_lib

Run the `start_runtime_with_sample_lib` python script:

```shell
python3 -m examples.sample_lib.start_runtime_with_sample_lib
```

This will setup a config with both `grpc` and `http` servers enabled for inference and training. The script then starts the `caikit runtime server`. While the server is running, you can see the generated proto files in a directory called `protos`. (They will be auto-deleted once you kill the server)

We generate 3 services total:
- A `train` service. The proto for this service is `protos/samplelibtrainingservice.proto`
- An `inference` service. The proto for this service is `protos/samplelibservice.proto`
- A `training management` service. The proto for this service is `protos/trainingmanagement.proto`

You can now leave the server running and open a new terminal to proceed with next steps to train a model, check its training status and send an inference request to your model.

(To kill the server, press Ctrl + C. This will remove the `protos` directory to clean up.)

## Interact using the python client

You can run the python client using:

```shell
python3 -m examples.sample_lib.client
```

The python client sends in requests to all 3 services that were mentioned above, printing the result from each request.

## Interact using terminal

You can also use `grpcurl` (for gRPC requests) or `curl` (for http requests) to send in commands one-by-one to all the 3 services that were mentioned above.

Note: `http` does not currently support `training management` APIs.
### To train a model

#### Using gRPC

In order to train a model via gRPC, we will use `grpcurl` and point the import-path to `protos` dir, then call one of the Train rpc's available in the `SampleLibTrainingService` (see `protos/caikit_sample_lib.proto` file generated above for all Train rpcs):

```shell
grpcurl -plaintext -import-path protos/ -proto caikit_sample_lib.proto -d '{"model_name": "my_model", "parameters": {"training_data": {"file": {"filename": "protos/sample.json"}}}}' localhost:8085 caikit_sample_lib.SampleLibTrainingService/SampleTaskSampleModuleTrain
```

You should receive a response similar to the below:

```shell
{
  "trainingId": "wTHxlsu:5bdb5949-4efa-4512-bbac-709cbf37c00e",
  "modelName": "my_model"
}
```

Copy the `trainingId` to use in next step.

#### Using HTTP

Docs coming soon...

### To check on training status for a training

#### Using gRPC

With a `trainingId`, you can get a training status via gRPC. Replace the command below with your `trainingId`.

```shell
grpcurl -plaintext -import-path protos/ -proto caikit.runtime.training.proto -d '{"training_id": "<training_id>"}' localhost:8085 caikit.runtime.training.TrainingManagement/GetTrainingStatus
```

You should get a response like this:

```shell
{
  "trainingId": "wTHxlsu:5bdb5949-4efa-4512-bbac-709cbf37c00e",
  "state": "COMPLETED",
  "submissionTimestamp": "2023-08-30T22:19:13.739694Z",
  "completionTimestamp": "2023-08-30T22:19:13.744542Z"
}
```
Once your training is completed, you can proceed to call inference on the model.

#### Using HTTP

`http` currently doesn't support training status APIs. Coming soon...

### To call inference on a model with model Id

You are now ready to call inference via either gRPC or REST.

#### To use the gRPC Server for inference

You can also use the gRPC Server to call inference on this model by running:
```shell
grpcurl -plaintext -import-path protos/ -proto caikit_sample_lib.proto -d '{"sample_input": {"name": "world"}}' -H 'mm-model-id: my_model' localhost:8085 caikit_sample_lib.SampleLibService/SampleTaskPredict
```

You should receive a successful response back with a response body:
```shell
{
  "greeting": "Hello world"
}
```

#### To use the REST Server for inference

- In a browser of choice, visit `http://localhost:8080/docs/`. All the available inference rpcs are listed. Expand on the correct task for the model you trained. In this example, we are using `api/v1/{model_id}/task/sample`.
- Click "Try It Out"
- Fill in model_id "my_model" as used in the train a model step. Change the request body to your liking. Then click "Execute". Ex:

```shell
curl -X 'POST' \
  'http://localhost:8080/api/v1/task/sample' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"inputs": {"name": "world"}, "model_id": "my_model"}'
```

You should receive a 200 response back with a response body:
```shell
{
  "greeting": "Hello IBM"
}
```

## Interact using a combination of pb2s and DataModels

Install `protoc`,

```shell
pip3 install grpcio-tools
``````

then generate the compiled `pb2` files,

```shell
python3 -m grpc_tools.protoc -I protos --python_out=examples/sample_lib/generated --grpc_python_out=examples/sample_lib/generated protos/*.proto
```

then start the client using the `pb2` files along with the `sample_lib` DataModel:

```shell
python3 -m examples.sample_lib.client_proto
```