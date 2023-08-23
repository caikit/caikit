
# Interacting with the Sample lib

Run `python3 -m examples.sample_lib.start_runtime_with_sample_lib` in one terminal.

The following sections show how to interact with the server using either REST or gRPC.

## REST

Head over to `localhost:8080/docs`. First train a module then try inferencing on it. (Training management API is not yet supported on REST)

## GRPC

The server should automatically create a `protos` dir. CD into that, `cd protos`, then try the following commands.

### Training

```grpcurl -plaintext -proto samplelibtrainingservice.proto -d '{"model_name": "my_model", "training_data": {"file": {"filename": "protos/sample.json"}}}' localhost:8085 caikit.runtime.SampleLib.SampleLibTrainingService/SampleTaskSampleModuleTrain```

### Training status

```grpcurl -plaintext -proto trainingmanagement.proto -d '{"training_id": "<training_id"}' localhost:8085 caikit.runtime.training.TrainingManagement/GetTrainingStatus```

### Inference

```grpcurl -plaintext -proto samplelibservice.proto -d '{"sample_input": {"name": "blah"}}' -H 'mm-model-id: my_model' localhost:8085 caikit.runtime.SampleLib.SampleLibService/SampleTaskPredict```