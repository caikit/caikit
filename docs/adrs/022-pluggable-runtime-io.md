# ADR N022: Pluggable Runtime IO

Some users of `caikit` will require application-specific logic to fetch data from secure locations. These implementations will not be sufficiently generic to be maintained in the core of `caikit`, but for these security-minded applications, they are a must-have in order to satisfy security posture that fits the existing application.

## Decision

This proposal is to make the Input/Output mechanisms for `caikit.runtime` pluggable. Currently, this means updating `DataStreamSource` and the current `output_path` mechanism to use `oneof` semantics that can be extended by new implementations and bound at boot time with config options.

## Status

choose one: Accepted

## Consequences

* New `config` section `data_streams.source_plugins` that will control which data stream source types are supported in the running `caikit` environment
* Equivalent section for `output_path` (naming TBD)
* An API breaking change to change `output_path` to something like `output_target` to be more generic and support a `oneof` for different output target representations
* New base classes and factories to register user-defined implementations of the plugins
