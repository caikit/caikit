
# Interacting with the Sample lib

This document describes how to quickly get a runtime server built with `sample_lib` library, train a model with gRPC and with that trained model, send an inference call to the server with either HTTP or gRPC call.

## Build and start a runtime server with sample_lib

Run the `start_runtime_with_sample_lib` python script:

```shell
python3 -m examples.sample_lib.start_runtime_with_sample_lib
```

This will setup a config with both `grpc` and `http` servers enabled for inference service, as well as training service. The library used with some sample modules is `sample_lib`. The script then starts the caikit runtime server. While the server is running, you can see the generated protobufs in a directory called `protos`. 

We generate 3 services total:
- A train service. The proto for this service is `protos/samplelibtrainingservice.proto`
- An inference service. The proto for this service is `protos/samplelibservice.proto`
- A training management service. The proto for this service is `protos/trainingmanagement.proto`

You can now leave the server running and open a new terminal to proceed with next steps to train a model, check its training status and send an inference request to your model.

(To kill the server, press Ctrl + C. This will remove the `protos` directory to clean up.)

## To train a model

As of caikit v`0.18.0`, we do not yet support training model via REST.
In order to train a model via gRPC, we will use `grpcurl` and point the import-path to `protos` dir, then call one of the Train rpc's available in the `SampleLibTrainingService` (see `protos/samplelibtrainingservice.proto` file generated above for all Train rpcs):

```shell
grpcurl -plaintext -import-path protos/ -proto samplelibtrainingservice.proto -d '{"model_name": "my_model", "training_data": {"file": {"filename": "protos/sample.json"}}}' localhost:8085 caikit.runtime.SampleLib.SampleLibTrainingService/SampleTaskSampleModuleTrain
```

You should receive a response similar to the below:

```shell
{
  "trainingId": "wTHxlsu:5bdb5949-4efa-4512-bbac-709cbf37c00e",
  "modelName": "my_model"
}
```

Copy the `trainingId` to use in next step.

## To check on training status of a trainingId

With a `trainingId`, you can get a training status via gRPC. Replace the command below with your `trainingId`.

```shell
grpcurl -plaintext -import-path protos/ -proto trainingmanagement.proto -d '{"training_id": "<training_id"}' localhost:8085 caikit.runtime.training.TrainingManagement/GetTrainingStatus
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

## To call inference on a model with modelId

You are now ready to call inference via either gRPC or REST.

### To use the REST Server for inference

- In a browser of choice, visit `http://localhost:8080/docs/`. All the available inference rpcs are listed. Expand on the correct task for the model you trained. In this example, we are using `api/v1/{model_id}/task/sample`.
- Click "Try It Out"
- Fill in model_id "my_model" as used in the train a model step. Change the request body to your liking. Then click "Execute". Ex:

```shell
curl -X 'POST' \
  'http://localhost:8080/api/v1/my_model/task/sample' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "inputs": {
    "name": "IBM"
  }
}'
```

You should receive a 200 response back with a response body:
```shell
{
  "greeting": "Hello IBM"
}
```

### To use the gRPC Server for inference

You can also use the gRPC Server to call inference on this model by running:
```shell
grpcurl -plaintext -import-path protos/ -proto samplelibservice.proto -d '{"sample_input": {"name": "IBM"}}' -H 'mm-model-id: my_model' localhost:8085 caikit.runtime.SampleLib.SampleLibService/SampleTaskPredict
```

You should receive a successful response back with a response body:
```shell
{
  "greeting": "Hello IBM"
}
```