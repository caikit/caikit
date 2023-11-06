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
Tests for the uniform health probe

🌶️🌶️🌶️ This test relies on test infrastructure in caikit.runtime, so the test
needs to live inside tests/runtime even though the functionality being tested is
not. If this is moved to the top of tests, the runtime test infra boots up too
early causing some of the core tests to fail!
"""
# Standard
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from unittest import mock
import os

# Third Party
import pytest
import tls_test_tools

# First Party
from caikit_health_probe import __main__ as caikit_health_probe
import alog

# Local
from caikit import get_config
from tests.conftest import temp_config
from tests.runtime.conftest import runtime_grpc_test_server, runtime_http_test_server
from tests.runtime.http_server.test_http_server import generate_tls_configs

## Helpers #####################################################################

log = alog.use_channel("TEST")


@contextmanager
def maybe_runtime_grpc_test_server(*args, **kwargs):
    if get_config().runtime.grpc.enabled:
        with runtime_grpc_test_server(*args, **kwargs) as grpc_server:
            yield grpc_server
    else:
        yield


@contextmanager
def maybe_runtime_http_test_server(*args, **kwargs):
    if get_config().runtime.http.enabled:
        with runtime_http_test_server(*args, **kwargs) as http_server:
            yield http_server
    else:
        yield


@contextmanager
def temp_probe_config(*args, **kwargs):
    with temp_config(*args, **kwargs) as the_config:
        get_config_mock = mock.MagicMock(return_value=the_config)
        with mock.patch.object(caikit_health_probe, "get_config", get_config_mock):
            yield


class TlsMode(Enum):
    INSECURE = 0
    TLS = 1
    MTLS = 3


class ServerMode(Enum):
    HTTP = 0
    GRPC = 1
    BOTH = 2


@dataclass
class ProbeTestConfig:
    tls_mode: TlsMode
    server_mode: ServerMode
    # TLS blobs passed as inline strings instead of files
    inline: bool = False
    # Run the unix socket grpc server
    unix_socket: bool = True
    # Put "localhost" in the SAN list for the server's cert
    localhost_in_cert: bool = True
    # Use a common CA for client and server certs (mTLS only)
    common_client_ca: bool = True
    # Whether the test should eventually become healthy
    should_become_healthy: bool = True


## Tests #######################################################################


@pytest.mark.parametrize(
    "test_config",
    [
        # Insecure
        ProbeTestConfig(TlsMode.INSECURE, ServerMode.HTTP),
        ProbeTestConfig(TlsMode.INSECURE, ServerMode.GRPC),
        ProbeTestConfig(TlsMode.INSECURE, ServerMode.BOTH),
        # TLS
        ProbeTestConfig(TlsMode.TLS, ServerMode.HTTP),
        ProbeTestConfig(TlsMode.TLS, ServerMode.GRPC),
        ProbeTestConfig(TlsMode.TLS, ServerMode.BOTH),
        ProbeTestConfig(TlsMode.TLS, ServerMode.BOTH, inline=True),
        ProbeTestConfig(TlsMode.TLS, ServerMode.BOTH, localhost_in_cert=False),
        # mTLS
        ProbeTestConfig(TlsMode.MTLS, ServerMode.HTTP),
        ProbeTestConfig(TlsMode.MTLS, ServerMode.GRPC),
        ProbeTestConfig(TlsMode.MTLS, ServerMode.BOTH),
        ProbeTestConfig(TlsMode.MTLS, ServerMode.BOTH, inline=True),
        ProbeTestConfig(TlsMode.MTLS, ServerMode.BOTH, localhost_in_cert=False),
        ProbeTestConfig(TlsMode.MTLS, ServerMode.BOTH, common_client_ca=False),
        # Invalid configs that never pass
        ProbeTestConfig(
            TlsMode.TLS,
            ServerMode.GRPC,
            localhost_in_cert=False,
            unix_socket=False,
            should_become_healthy=False,
        ),
        ProbeTestConfig(
            TlsMode.TLS,
            ServerMode.BOTH,
            localhost_in_cert=False,
            unix_socket=False,
            should_become_healthy=False,
        ),
        ProbeTestConfig(
            TlsMode.MTLS,
            ServerMode.GRPC,
            localhost_in_cert=False,
            unix_socket=False,
            should_become_healthy=False,
        ),
    ],
)
def test_health_probe(test_config: ProbeTestConfig):
    """Test all of the different ways that the servers could be running"""
    with alog.ContextLog(log.info, "---LOG CONFIG: %s---", test_config):
        # Get ports for both servers
        http_port = tls_test_tools.open_port()
        grpc_port = tls_test_tools.open_port()

        # Set up SAN lists if not putting "localhost" in
        server_sans, client_sans = None, None
        if not test_config.localhost_in_cert:
            server_sans = ["foo.bar"]
            client_sans = ["baz.bat"]

        # Set up tls values if needed
        with generate_tls_configs(
            port=http_port,
            tls=test_config.tls_mode == TlsMode.TLS,
            mtls=test_config.tls_mode == TlsMode.MTLS,
            inline=test_config.inline,
            separate_client_ca=not test_config.common_client_ca,
            server_sans=server_sans,
            client_sans=client_sans,
        ) as config_overrides:
            with temp_probe_config(
                {
                    "runtime": {
                        "grpc": {
                            "port": grpc_port,
                            "enabled": test_config.server_mode
                            in [ServerMode.GRPC, ServerMode.BOTH],
                            "unix_socket_path": os.path.join(
                                config_overrides["use_in_test"]["workdir"],
                                "grpc.sock",
                            )
                            if test_config.unix_socket
                            else None,
                        },
                        "http": {
                            "enabled": test_config.server_mode
                            in [ServerMode.HTTP, ServerMode.BOTH],
                        },
                    }
                },
                "merge",
            ):
                # Health probe fails with no servers booted
                assert not caikit_health_probe.health_probe()
                # If booting the gRPC server, do so
                with maybe_runtime_grpc_test_server(grpc_port):
                    # If only running gRPC, health probe should pass
                    assert caikit_health_probe.health_probe() == (
                        test_config.should_become_healthy
                        and test_config.server_mode == ServerMode.GRPC
                    )
                    # If booting the HTTP server, do so
                    with maybe_runtime_http_test_server(
                        http_port,
                        tls_config_override=config_overrides,
                        check_readiness=test_config.should_become_healthy,
                    ):
                        # Probe should always pass with both possible servers up
                        assert (
                            caikit_health_probe.health_probe()
                            == test_config.should_become_healthy
                        )
