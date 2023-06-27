# ADR 018: Shared Backends

The main value proposition for `caikit` is the ability to standardize interfaces for AI workloads (primarily `train` and `run`). Currently, the primary way to accomplish this is via different `Module` implementations for the same logical task where the workload runs locally in the python process using some backend machine learning library (e.g. `torch`). The `module_backends` framework allows individual `Module` implementations to require access to a globally configured instance of a `BackendBase` that holds connection details to a non-local runtime engine.

Currently, all uses of a `module_backend` require that the `Module` implementation have explicit knowledge of the backend that will be used to run it so that it can call the backend's functionality directly in `train` and/or `run`. This is necessary when a given backend can only perform handle a subset of workloads (e.g. running distributed `torch` training jobs). There are, however, a collection of logical backends which are capable of running a `Module` as a black box. Some examples include:

* Generic training / inference performed using [`ray`](https://www.ray.io/)
* Proxy training / inference delegating calls to remote scaled instances of `caikit.runtime` running elsewhere in a cluster

## Decision

In order to support generic black-box backends, this ADR introduces two new abstractions: `SharedTrainBackend` and `SharedLoadBackend`. In addition to [the abstract interface of `BackendBase`](https://github.com/caikit/caikit/blob/main/caikit/core/module_backends/base.py), these backends must provide functionality to take a reference to a `Module` class and passthrough arguments to `train` and `load` respectively which will wrap up the semantics of executing the `train`/`load`/`run` semantics in the target backend platform.

One particular decision here is that the responsibility for delegating `inference` jobs is managed through a `load` backend rather than a `run` backend. The insight is that inference requires loaded model state in the backend engine, so a `SharedLoadBackend` is responsible for returning a `Module` instance whose `run` function will invoke the target module in the backend infrastructure.


## Status

choose one: [Superseded by 020-model-management-abstractions](020-model-management-abstractions.md)

## Consequences

There are several consequences of this ADR:

1. New backend infrastructure can be built that perform `load` and `train` operations against a given remote backend relying only on the interface of `ModuleBase`.
2. The `LOCAL` backend can be refactored as a `SharedTrainBackend` and a `SharedLoadBackend`