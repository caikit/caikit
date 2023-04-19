# ADR 10: Data model definition

The Caikit data model is designed to encapsulate all information about a concrete model in one atomic component. Data models are designs of how various elements of data are organized and how they relate to one another.

Most open source frameworks such as [Tensorflow](https://www.tensorflow.org/) use an API that outputs the lowest common denominator between all tasks, such as a tensor in the mathematical space. This means that the framework client has to be aware of translation from the semantic to the mathematical space, and vice versa.

Even in frameworks that provide adapter functionality like [KServe](https://kserve.github.io/website/0.10/), adapters have to live in process space, generally involving a more complicated architectural topology to understand and maintain. Caikit instead has the translation logic between the semantic and mathematical spaces in the data model atomic logic.

## Decision

We will use data models to capture the semantic inputs and outputs of AI tasks.


## Status

Accepted


## Consequences

- Caikit users will be able to access model information through data models
- Caikit users may face one extra level of indirection if they want to access mathematical representations of their data models results (e.g. arrays).
- Caikit will provide the same data model per AI task [TODO: link to task ADR].
- Data models will be backed by a common backend [TODO: link to backends, protobuf ADRs].
- AI model authors will have to translate model inputs and outputs to work with data models. This has implications on usability and stability.
