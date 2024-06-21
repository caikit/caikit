"""
Unit tests for the trace module
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Optional
from unittest import mock
import builtins
import ssl
import sys
import threading
import time

# Third Party
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
import grpc
import pytest
import uvicorn

# Local
from caikit.runtime import trace
from tests.conftest import temp_config
from tests.fixtures import Fixtures
from tests.runtime.conftest import generate_tls_configs, open_port, reset_trace

## Mock Collectors #############################################################


class MockCollectorHttpServer:
    """Mock http server implementing collection"""

    def __init__(
        self,
        port: int,
        cert: Optional[str] = None,
        key: Optional[str] = None,
    ):
        self.requests = []
        self.app = FastAPI()

        @self.app.post("/v1/traces")
        def traces(request: Request):
            self.requests.append(request)
            return PlainTextResponse("OK")

        tls_kwargs = {}
        if cert and key:
            tls_kwargs["ssl_keyfile"] = key
            tls_kwargs["ssl_certfile"] = cert
        self.server = uvicorn.Server(
            uvicorn.Config(
                self.app, port=port, timeout_graceful_shutdown=0.001, **tls_kwargs
            )
        )
        self.server_thread = threading.Thread(target=self.server.run)

    def start(self):
        self.server_thread.start()
        while not self.server.started:
            time.sleep(1e-3)

    def stop(self):
        self.server.should_exit = True
        self.server_thread.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_, **__):
        self.stop()


@contextmanager
def collector_grpc(
    port: int,
    cert: Optional[str] = None,
    key: Optional[str] = None,
    client_ca: Optional[str] = None,
):
    """Define and instantiate a collector grpc server

    NOTE: The classes themselves are defined inline so that the imports are
        scoped to the test
    """

    # Third Party
    from opentelemetry.proto.collector.trace.v1 import (
        trace_service_pb2,
        trace_service_pb2_grpc,
    )

    class MockServicer(trace_service_pb2_grpc.TraceServiceServicer):
        def __init__(self):
            self.requests = []

        def Export(self, request, *_, **__):
            self.requests.append(request)
            return trace_service_pb2.ExportTraceServiceResponse()

    # Set up the servicer and server
    servicer = MockServicer()
    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    trace_service_pb2_grpc.add_TraceServiceServicer_to_server(servicer, server)

    # Bind to the port and start up
    server_hname = f"[::]:{port}"
    if cert and key:
        tls_pair = [(key.encode("utf-8"), cert.encode("utf-8"))]
        if client_ca:
            creds = grpc.ssl_server_credentials(
                tls_pair,
                root_certificates=client_ca.encode("utf-8"),
                require_client_auth=True,
            )
        else:
            creds = grpc.ssl_server_credentials(tls_pair)
        server.add_secure_port(server_hname, creds)
    else:
        server.add_insecure_port(server_hname)
    server.start()

    # Yield the servicer for checking requests
    try:
        yield servicer
    finally:
        server.stop(0)


def maybe_inline(inline: bool, tls_file: str):
    if not inline:
        return tls_file
    with open(tls_file, "r") as handle:
        return handle.read()


## Fixtures ####################################################################


@contextmanager
def trace_config(**kwargs):
    with temp_config({"runtime": {"trace": kwargs}}, "merge"):
        yield


@pytest.fixture
def trace_enabled_http():
    with trace_config(
        enabled=True, protocol="http", endpoint="http://localhost:1234/v1/traces"
    ):
        yield


@pytest.fixture
def trace_enabled_grpc():
    with trace_config(enabled=True, protocol="grpc"):
        yield


@pytest.fixture
def trace_disabled():
    with temp_config({"runtime": {"trace": {"enabled": False}}}, "merge"):
        yield


@pytest.fixture
def collector_grpc_insecure(open_port):
    with collector_grpc(open_port) as servicer_mock:
        with trace_config(endpoint=f"localhost:{open_port}"):
            yield servicer_mock


@pytest.fixture
def collector_http_insecure(open_port):
    with MockCollectorHttpServer(open_port) as server_mock:
        with trace_config(endpoint=f"http://localhost:{open_port}/v1/traces"):
            yield server_mock


@pytest.fixture
def reset_otel_trace_globals():
    """https://github.com/open-telemetry/opentelemetry-python/blob/main/tests/opentelemetry-test-utils/src/opentelemetry/test/globals_test.py#L25"""
    # Third Party
    from opentelemetry import trace as otel_trace
    from opentelemetry.util._once import Once

    with mock.patch.object(otel_trace, "_TRACER_PROVIDER_SET_ONCE", Once()):
        with mock.patch.object(otel_trace, "_TRACER_PROVIDER", None):
            yield


## Helpers #####################################################################


def exercise_tracer_api(tracer):
    """Shared helper to exercise the full scope of the Tracer that we expect to
    support
    """
    with tracer.start_as_current_span("foobar") as span:
        span.set_attribute("foo", "bar")
        span.set_attributes({"baz": "bat", "biz": 123})
        span.add_event("something", {"key": ["val"]})

        # NOTE: Just in case anyone finds this later. The `context` arg of
        #   start_span needs to be an opentelemetry.context.Context (which is a
        #   dict) NOT a SpanContext (which is not a dict), so you cannot call
        #   start_span(...) with context=span.get_span_context()!
        nested_span1 = tracer.start_span("nested1", links=[span])
        nested_span2 = tracer.start_span("nested2", links=[span])
        nested_span1.add_link(nested_span2.get_span_context())


def verify_exported(mock_server):
    """Verify that output was exported to the mock server"""
    # Third Party
    from opentelemetry.trace import get_tracer_provider

    get_tracer_provider().force_flush()
    assert mock_server.requests


## Tests #######################################################################


def test_trace_unconfigured(reset_trace, trace_disabled):
    """Test that without calling configure, all of the expected tracing
    operations are no-ops
    """
    exercise_tracer_api(trace.get_tracer("test/tracer"))
    assert "opentelemetry" not in sys.modules


def test_trace_disabled(reset_trace, trace_disabled):
    """Test that with configure called, but trace disabled, all of the expected
    tracing operations are no-ops
    """
    trace.configure()
    exercise_tracer_api(trace.get_tracer("test/tracer"))
    assert "opentelemetry" not in sys.modules


def test_trace_not_installed(reset_trace, trace_enabled_grpc):
    """Test that when the libraries cannot be imported, the configure step does
    not raise
    """

    with mock.patch.object(builtins, "__import__", side_effect=ImportError("yikes")):
        trace.configure()


def test_trace_configured_grpc(
    reset_otel_trace_globals, trace_enabled_grpc, collector_grpc_insecure
):
    """Test that with tracing enabled using the grpc protocol, all of the
    expected tracing operations are correctly configured and run
    """
    with trace_config(flush_on_exit=False):
        trace.configure()
    exercise_tracer_api(trace.get_tracer("test/tracer"))
    assert "opentelemetry" in sys.modules
    verify_exported(collector_grpc_insecure)


def test_trace_configured_http(
    reset_otel_trace_globals, trace_enabled_http, collector_http_insecure
):
    """Test that with tracing enabled using the http protocol, all of the
    expected tracing operations are correctly configured and run
    """
    with trace_config(flush_on_exit=False):
        trace.configure()
    exercise_tracer_api(trace.get_tracer("test/tracer"))
    assert "opentelemetry" in sys.modules
    verify_exported(collector_http_insecure)


@pytest.mark.parametrize(
    ["mtls", "inline"],
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_trace_grpc_tls(
    reset_otel_trace_globals, trace_enabled_grpc, open_port, mtls, inline
):
    """Test that tracing can be enabled all flavors of (m)TLS"""
    with generate_tls_configs(
        open_port,
        tls=True,
        mtls=mtls,
        inline=True,
    ) as tls_configs:
        mtls_kwargs = (
            {
                "client_ca": maybe_inline(True, tls_configs.use_in_test.ca_cert),
            }
            if mtls
            else {}
        )
        with collector_grpc(
            open_port,
            cert=tls_configs.runtime.tls.server.cert,
            key=tls_configs.runtime.tls.server.key,
            **mtls_kwargs,
        ) as servicer:
            tls_trace_cfg = {
                "ca": maybe_inline(inline, tls_configs.use_in_test.ca_cert)
            }
            if mtls:
                tls_trace_cfg["client_cert"] = maybe_inline(
                    inline, tls_configs.use_in_test.client_cert
                )
                tls_trace_cfg["client_key"] = maybe_inline(
                    inline, tls_configs.use_in_test.client_key
                )
            with trace_config(
                tls=tls_trace_cfg,
                flush_on_exit=False,
                endpoint=f"localhost:{open_port}",
            ):
                trace.configure()
            exercise_tracer_api(trace.get_tracer("test/tracer"))
            verify_exported(servicer)


def test_trace_http_tls(reset_otel_trace_globals, trace_enabled_http, open_port):
    """Test that tracing can be enabled all flavors of (m)TLS"""
    with generate_tls_configs(open_port, tls=True, inline=False) as tls_configs:
        with MockCollectorHttpServer(
            open_port,
            cert=tls_configs.runtime.tls.server.cert,
            key=tls_configs.runtime.tls.server.key,
        ) as servicer:
            with trace_config(
                tls={"ca": tls_configs.use_in_test.ca_cert},
                flush_on_exit=False,
                endpoint=f"https://localhost:{open_port}/v1/traces",
            ):
                trace.configure()
            exercise_tracer_api(trace.get_tracer("test/tracer"))
            verify_exported(servicer)


@pytest.mark.parametrize(
    ["context", "configure", "should_return"],
    [
        (Fixtures.build_context(), True, True),
        (Request({"type": "http", "headers": {}}), True, True),
        (Fixtures.build_context(), False, False),
        (Request({"type": "http", "headers": {}}), False, False),
        (None, True, False),
    ],
)
def test_trace_get_trace_context(context, configure, should_return, reset_trace):
    """Test that get_trace_context returns a context under the right
    circumstances
    """
    with trace_config(enabled=True):
        if configure:
            trace.configure()
        ctx = trace.get_trace_context(context)
        assert ((ctx is not None) and should_return) or (
            ctx is None and not should_return
        )
