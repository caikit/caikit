### Context

Up until now, caikit has primarily managed its server runtime using gRPC. This decision was made during the early stages of the project based on the initial usecases that gave rise to the project in the first place. The decision to use gRPC brought the advantages of strict APIs with protobuf and auto-generated server/client stubs. The main downside of the gRPC decision has been that as a technology it's less well known than REST and the developer experience has a steeper learning curve.

As caikit moves to being a general-purpose framework for managing production-grade model definitions and operations, supporting a native REST server has become increasingly critical to enable the "15-minutes to value" experience. Prior to this ADR, REST has been supported using grpc-gateway-wrapper which has not been sufficiently flexible for all usecases (including streaming).

Finally, the most popular open model hosting framework at the time of this ADR is huggingface (https://huggingface.co/). HF offers a [REST API](huggingface.co/docs/api-inference/detailed_parameters) for its task-specific inference endpoints. The current consumers of caikit are strategically aligned with HF for open-model collaboration, and therefore caikit needs to provide aligned APIs with the HF task inference endpoints.

### Decision

* Caikit will provide a HTTP server that serves REST (unary) and Server Side Events (streaming) endpoints analogously to how the gRPC server provides RPCs based on task module `run` and `train` functions
* Caikit's HTTP API will frame its input data structures with "inputs" and "parameters" to follow how the HF APIs delineate between data inputs and algorithmic parameters