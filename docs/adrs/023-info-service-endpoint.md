# ADR N023: Info Service Endpoint

Some users of `caikit` find it useful to be able to easily retrieve versioning information to help with debugging. This includes details on the version of caikit libraries as well as the runtime image.

## Decision

This proposal is to add an info servicer in order to create an endpoint in both the gRPC and HTTP server in order to get version information. The info servicer will be generic so it can be used for other information requests in the future such as model management metadata. This will return a dictionary of version information for caikit libraries by default and additionally python packages and the runtime image if set.

For HTTP the endpoint will be something like `/info/version` and for gRPC it will be something like `GetRuntimeInfo()`.

**Example Config**
```yaml
runtime:
    version_info:
        python_packages:
            all: false
        runtime_image: ""
```

## Status

choose one: Accepted

## Consequences

* Users can hit an info endpoint that will provide caikit versioning details as well as versions for other python packages.
* New `config` section `runtime.version_info` that will control what version information is retrieved to add to the endpoint.
