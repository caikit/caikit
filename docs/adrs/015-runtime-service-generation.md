# ADR 15: Generate Services to Run Models Anywhere

One of the main goals of caikit is portability- write your model code once and deploy it to run anywhere.
Achieving this portability requires stable network interfaces to invoke the models remotely.

Model authors already hold the burden of defining a usable python API for their models.
Requiring humans to define yet another interface for a web API is onerous, error-prone, and would require 
yet another layer of translation between the web API and the python API.

We would also like a remote invocation of a model to look and feel like a local invocation of a model.
Users should be able to write applications that run models locally, and easily offload the model computation
to a remote service depending on their deployment context, with no code changes required. Having 
separately-authored remote APIs would make this hard, if not impossible to achieve.

Module interfaces should already be defined using parameters that are easily translatable to protobuf:
python primitives and [data models](010-data-model-definition.md).

The protobuf stack for data models is already managed in-memory (as opposed to authoring and compiling 
protobuf files.) It's not a stretch to also manage protobuf services and some gRPC bits and pieces in memory
as well.

## Decision

We will translate the python API of models' `train` and `run` functions to pure protobuf APIs
and generate (a) gRPC service(s) for the user's library.

## Status

Accepted

## Consequences

- Network interfaces will be available to users without any extra effort
  - Except that python is not a typed language, so users will have to adopt one of a few available standard practices (type annotations or docstrings formatted with a known standard) to denote the type of each parameter in their `run` or `train` methods.
- The burden of generating valid service interfaces may place restrictions on how flexible the user's definitions of data models may be
- Service generation code is more code we need to maintain, and translating from python apis is reasonably complex
- "Power user" features like explicitly specifying field ordering in order to ensure API backwards-compatibility as a library evolves will be more difficult to plumb into an api generator
- Caikit has an easy path forward for automating remote module backends
