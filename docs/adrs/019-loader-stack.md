# ADR 019: Loader Stack

When loading a model in a non-local backend, there are several ways to map a given set of model artifacts to a concrete `ModuleBase` derived class that can perform the `load` operation. With the introduction of [shared load backends](./018-shared-backends.md), we could either require that _any_ shared backend be able to load _all_ models, or we could require that _some_ shared backend be able to load _any_ model. This ADR proposes that we do the latter in order to support a mixed set of loaders that can combine to support the set of modules a given usage requires.

## Decision

* When executing the global `load` logic, each `SharedLoadBackend` may optionally return `None` to indicate failure to load the given model
* The stack of load backends will be iterated in priority order
* As soon as a load backend succeeds in loading a model, the model is considered loaded
* If no load backend is able to load a model, an error is raised


## Status

choose one: [Superseded by 020-model-management-abstractions](020-model-management-abstractions.md)

## Consequences

1. `SharedLoadBackend` classes must support _some_ modules, but do not need to support _all_ modules
2. The configuration framework must allow for multiple load backends to be configured in priority order