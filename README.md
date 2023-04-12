# Caikit

Caikit is an AI toolkit that enables users to manage models through a set of developer friendly APIs. It provides a consistent format for creating and using AI models against a wide variety of data domains and tasks.

![Caikit Overview](https://raw.githubusercontent.com/caikit/caikit/main/caikit-overview.png)

## Capabilities

Caikit streamlines the management of AI models for application usage by letting AI model authors focus on solving well known problems with novel technology. With a set of model implementations based on Caikit, you can:

- Run training jobs to create models from your data
- Run model inference using data APIs that represent data as structures rather than tensors
- Implement the right training techniques to fit the model, from static regexes to multi-GPU distribution
- Merge models from diverse AI communities into a common API (e.g. `transformers`, `tensorflow`, `sklearn`, etc...)
- Update applications to newer models for a given task without client-side changes

## What Differentiates Caikit from Other AI Model Runtimes?

Developers who write applications that consume AI models are not necessarily AI experts who understand the intricate details of the AI models that they use. Some would like to treat AI as a "black box function" where they give it input and it returns the output. This is similar in cloud computing whereby some users would like to deploy their applications to the cloud without detailed knowledge of the cloud infrastructure. The value for them is in their application and that is what is of most interest to them.

Caikit provides an abstraction layer for application developers where they can consume AI models through APIs independent of understanding the data form of the model. In other words, the input and output to the model is in a format which is easily programmable and does not require data transformations. This facilitates the model and the application to evolve independently of each other.

When deploying a small handful of models, this benefit is minimal. The benefits are generally realized when consuming 10s or hundreds of AI models, or maintaining an application over time as AI technology evolves. Caikit simplifies the scaling and maintenance of such integrations compared to other runtimes. This is because other runtimes require an AI centric view of the model (for example, the common interface of “tensor in, tensor out”) which means having to code different data transformations into the application for each model. Additionally, the data form of the model may change from version to version.

## User Profiles

There are 2 user profiles who leverage Caikit:

- AI Model Author:
  - Model Authors build and train AI models for data analysis
  - They bring data and tuning params to a pre-existing model architecture and create a new concrete model using APIs provided by Caikit
  - Examples of model authors are machine learning engineers, data scientists, and AI developers
- AI Model Operator:
  - Model operators use an existing AI model to perform a specific function within the context of an application
  - They take trained models, deploy them, and then infer the models in applications through APIs provided by Caikit
  - Examples of operators are cloud and embedded application developers whose applications need analysis of unstructured data

## AI Model Operator Quick Start

**UNDER CONSTRUCTION**

At this point, Caikit does not publish any pre-defined task implementations that can be readily consumed. Stay tuned!


### Prerequisitive

Install Caikit toolkit:

```console
pip install caikit
pip install caikit-nlp

# TODO: import the libraries to enable the sample code to run
```

### Preprocessing

Start by loading training data from Consumer Financial complaint database:

**TODO: show how to preprocess sample training data**

```python

```

### Training

Train an TF-IDF SVM classifier model on the training data:

```python
import caikit_nlp

syntax_model = caikit.load('syntax_izumo_en_stock'))

tfidf_classification_model = caikit_nlp.workflows.classification.base_classifier.tfidf_svm.TFidfSvm.train(
    training_data=training_data,
    syntax_model=syntax_model, 
    tfidf_svm_epochs=1,
    multi_label=True
)
```

### Inference

Use the trained model for prediction on new data:

```python
tfidf_svm_preds = tfidf_classification_model.run(text)
```

## Start the runtime with the sample_lib! 

Useful for interacting with the runtime with the sample_lib and playing around with it.

### Start the server
In one terminal, run the script that starts the server:

```
python3 -m examples.start_runtime_with_sample_lib
```

### Start training a sample block

This script will also create `protos`. In order to interact with the `sample_lib`, in another terminal, go inside that directory:

```
cd protos
```
then

```
grpcurl -plaintext -proto samplelibtrainingservice.proto -d '{"model_name": "my_model", "training_data": {"file": {"filename": "protos/sample.json"}}}' localhost:8085 caikit.runtime.SampleLib.SampleLibTrainingService/BlocksSampleTaskSampleBlockTrain
```
### Fetch status through training management API

```
grpcurl -plaintext -proto trainingmanagement.proto -d '{
    "training_id": "<training_id_from_above_response>"
}' localhost:8085 caikit.runtime.training.TrainingManagement/GetTrainingStatus
```
### Run prediction on trained block
```
grpcurl -plaintext -proto samplelibservice.proto -d '{"sample_input": {"name": "blah"}}' -H 'mm-model-id: my_model' localhost:8085 caikit.runtime.SampleLib.SampleLibService/SampleTaskPredict
```

</strike>

## Code of conduct

Participation in the Caikit community is governed by the [Code of Conduct](CODE-OF-CONDUCT.md).
