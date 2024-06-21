# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The trace module holds utilities for tracing runtime requests.
"""
# Standard
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterable, Optional, Union
import os

# Third Party
import grpc

# First Party
import alog

# Local
from ..config import get_config
from ..core.data_model.runtime_context import RuntimeServerContextType
from ..core.exceptions import error_handler

log = alog.use_channel("TRACE")
error = error_handler.get(log)


# Global handle to the trace and propagate modules that will be populated in
# configure()
_TRACE_MODULE = None
_PROPAGATE_MODULE = None


if TYPE_CHECKING:
    # Third Party
    from opentelemetry import Context
    from opentelemetry.trace import Span, Tracer


def configure():
    """Configure all tracing based on config and installed packages"""
    global _TRACE_MODULE
    global _PROPAGATE_MODULE

    # Short circuit if not enabled, including resetting the global module
    # pointer so that toggling from enabled -> disabled works as expected
    trace_cfg = get_config().runtime.trace
    if not trace_cfg.enabled:
        log.info("Trace disabled")
        _TRACE_MODULE = None
        return

    # Figure out which protocol is being used
    error.value_check("<RUN85736759E>", trace_cfg.protocol in ["grpc", "http"])
    grpc_protocol = trace_cfg.protocol == "grpc"

    # Attempt to import the necessary packages
    try:
        # Third Party
        from opentelemetry import propagate, trace
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Import the right span exporter
        if grpc_protocol:
            # Third Party
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
        else:
            # Third Party
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

    except ImportError as err:
        log.warning(
            "<RUN41027815W>",
            "Cannot enable trace. You may need to `pip install caikt[runtime-trace]`: %s",
            err,
            exc_info=True,
        )
        return

    # Populate the global module handle
    _TRACE_MODULE = trace
    _PROPAGATE_MODULE = propagate

    # Set up the exporter
    exporter_kwargs = {"endpoint": trace_cfg.endpoint}
    if trace_cfg.tls.ca:
        if grpc_protocol:
            creds_kwargs = {"root_certificates": _load_tls_secret(trace_cfg.tls.ca)}
            if trace_cfg.tls.client_key and trace_cfg.tls.client_cert:
                log.debug("Configuring grpc trace with mTLS")
                creds_kwargs["private_key"] = _load_tls_secret(trace_cfg.tls.client_key)
                creds_kwargs["certificate_chain"] = _load_tls_secret(
                    trace_cfg.tls.client_cert
                )
            else:
                log.debug("Configuring grpc trace with TLS")
            exporter_kwargs["credentials"] = grpc.ssl_channel_credentials(
                **creds_kwargs
            )
        else:
            error.value_check(
                "<RUN46942098E>",
                not (trace_cfg.tls.client_key and trace_cfg.tls.client_cert),
                "mTLS not supported for trace with HTTP",
            )
            log.debug("Configuring http trace with TLS")
            error.file_check("<RUN80171155E>", trace_cfg.tls.ca)
            exporter_kwargs["certificate_file"] = trace_cfg.tls.ca
    else:
        log.debug("Configuring trace with insecure transport")
        if grpc_protocol:
            exporter_kwargs["insecure"] = True
    exporter = OTLPSpanExporter(**exporter_kwargs)

    # Configure the trace provider
    resource = Resource(attributes={SERVICE_NAME: trace_cfg.service_name})
    provider = TracerProvider(
        resource=resource, shutdown_on_exit=trace_cfg.flush_on_exit
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def get_tracer(name: str) -> Union["_NoOpProxy", "Tracer"]:
    """Get a tracer that can be called with the opentelemetry API. If not
    configured, this will be a No-Op Proxy.
    """
    if _TRACE_MODULE:
        return _TRACE_MODULE.get_tracer(name)
    return _NoOpProxy()


def get_trace_context(runtime_context: RuntimeServerContextType) -> Optional["Context"]:
    """Extract the trace context from the runtime request context"""
    if runtime_context is None or not _PROPAGATE_MODULE:
        return None

    if isinstance(runtime_context, grpc.ServicerContext):
        return _PROPAGATE_MODULE.extract(
            carrier=dict(runtime_context.invocation_metadata())
        )

    # Local import of fastapi as an optional dependency
    try:
        # Third Party
        import fastapi

        if isinstance(runtime_context, fastapi.Request):
            return _PROPAGATE_MODULE.extract(carrier=runtime_context.headers)
    except ImportError:
        pass

    log.debug("Unknown context type: %s", type(runtime_context))
    return None


def set_tracer(runtime_context: RuntimeServerContextType, tracer: "Tracer"):
    """Helper to decorate a runtime context with a tracer if enabled"""
    if runtime_context:
        setattr(runtime_context, _CONTEXT_TRACER_ATTR, tracer)


@contextmanager
def start_child_span(
    runtime_context: RuntimeServerContextType,
    span_name: str,
) -> Iterable[Union["Span", "_NoOpProxy"]]:
    """Context manager that wraps start_as_current_span if enabled and tries to
    fetch a parent span from the runtime context
    """
    if (parent_tracer := getattr(runtime_context, _CONTEXT_TRACER_ATTR, None)) is None:
        parent_tracer = get_tracer(span_name)
    with parent_tracer.start_as_current_span(span_name) as span:
        yield span


## Implementation Details ######################################################

_CONTEXT_TRACER_ATTR = "__tracer__"


def _load_tls_secret(tls_config_val: str) -> bytes:
    """If the config value points at a file, load it, otherwise assume it's an
    inline string
    """
    if os.path.exists(tls_config_val):
        with open(tls_config_val, "rb") as handle:
            return handle.read()
    return tls_config_val.encode("utf-8")


class _NoOpProxy:
    """This dummy class is infinitely callable and will return itself on any
    getattr call or context enter/exit. It can be used to provide a no-op
    stand-in for all of the classes in the opentelemetry ecosystem when they are
    either not configured or not available.
    """

    def __getattr__(self, *_, **__):
        return self

    def __call__(self, *_, **__) -> "_NoOpProxy":
        return self

    def __enter__(self, *_, **__) -> "_NoOpProxy":
        return self

    def __exit__(self, *_, **__):
        pass
