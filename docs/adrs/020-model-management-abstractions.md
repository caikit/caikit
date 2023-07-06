# ADR 020: Model Management Abstractions

When managing models, there are three (and possibly eventually more) central abstract tasks:

* `training`: Take raw materials (including pretrained models) and create usable concrete model artifacts
* `finding`: Given some unique identifier for a concrete model, identify how the model should be loaded in `caikit`
* `initialization`: Given the information about a model that can be loaded in `caikit`, prepare the artifacts for online inference

The default implementation of each of these tasks is to perform them locally in the python process using the standard `caikit` model format, but each could be implemented using alternate technology. Here are some examples:

* `training`:
    * Train in a remote `ray` cluster where the `def train` function is executed as a `ray` job
    * Train from a local `caikit` library python process using a hosted training API in a running instance of `caikit.runtime` in a remote kubernetes cluster
* `finding`:
    * Find models in an alternate saving format such as `transformers`
    * Find models hosted in a remote storage service such as S3
* `initializing`:
    * Load the artifacts using `ray` serving and proxy `run` calls to the instance running in the ray cluster
    * Delegate the model to a hosted `caikit.runtime` instance in a remote cluster

The previous ADRs for [018-shared-backends](018-shared-backends.md) and [019-loader-stack](019-loader-stack.md) had several key problems:

* They attempted to solve these problems by merging them with the `module_backend` abstraction which has caused a lot of confusion since it was adopted.
* They didn't differentiate between `finding` and `initialization`
* They relied on an ordered list of configured objects to discover the right loader instance which was hard to configure in a running server since lists are not well supported with environment variables
    * This also did not allow explicit specification of a loader for a model where mulitple loaders in the list could work, so you could not load one model one way and another a different way

Prior to `018` and `019`, the `module_backend` abstraction was explicitly tasked with performing "internal distribution" (distributing the work of a given module to some framework, requiring specific knowledge of the algorithm implemented by the module). With `018` and `019`, `module_backend` was extended to also support "external distribution" (distributing the whole operation of a given module as a black box without knowledge of the module's algorithm). This was at the heart of what lead to the confusion, so in this ADR we walk that back and split those responsibilities between `module_backend` (responsible for "internal distribution") and `initializer` (responsible for "external distribution").

## Decision

* The `SharedTrainBackend` an `SharedLoadBackend` abstract classes will be removed
* The logic in `ModelManager` that handles `SharedLoadBackend`s will be removed
* A new set of abstract classes will be introduced in `caikit.core.model_management`, one each for `ModelTrainer`, `ModelFinder`, and `ModelLoader`
* The `ModelManager.load` implementation will be refactored to do the following:
    * Get a unique `finder` by name or value, falling back to a `"default"` instance and attempt to find the model based on `model_path`. A successful finder returns a `ModuleConfig` containing the `module_id` of the `module` class that should be used to initialize the model.
    * If the model is found, get a unique `initializer` by name or value, falling back to a `"default"` instance and attempt to initialize the found model using it.
* All of the responsibility for managing `module_backend` mapping will move to the `LOCAL` implementation of the `ModelLoader` abstraction
* The global `config` will be refactored as follows:
    * `module_backends` will be removed
    * A new section for `model_management` will be added with `trainers`, `finders`, and `initializers` under it
    * Each section will be a key/value map where the key is a unique string name and the value is a factory-constructible object with `"type"` and (optionally) `"config"` keys
    * The `backend_priority` logic will move to the `"config"` key of the `LOCAL` `initializer` configuration blob
    * Each section will be required to have a `"default"` key for the instance to use when none is passed by name
* (FUTURE) `ModelManager` will introduce a `train` function that uses the `ModelTrainer` abstraction in the same way that `load` uses the `ModelFinder`/`ModelLoader` abstractions
* Each abstraction will be managed using an extensible factory that can be extended in external libraries

## Status

choose one: Accepted


## Consequences

* New abstraction implementations can be added in external libraries to implement platform-specific logic (e.g. `S3BucketModuleFinder`)
* The confusion around `module_backends` will be removed and they will go back to being implementation details of specific modules
* Users will be able to explicitly name the `finder`/`initializer` to use for a given model without relying on global configuration
* The `config` API will break, causing churn with current user deployment topology
