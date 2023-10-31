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
"""
# Standard
from contextlib import contextmanager
from enum import Enum
from typing import Tuple
from unittest import mock

# Third Party
import pytest
import tls_test_tools

# First Party
from caikit_health_probe import __main__ as caikit_health_probe

# Local
from caikit import get_config
from tests.conftest import temp_config
from tests.runtime.conftest import runtime_grpc_test_server, runtime_http_test_server
from tests.runtime.http_server.test_http_server import generate_tls_configs

## Helpers #####################################################################


class TlsMode(Enum):
    INSECURE = 0
    TLS = 1
    MTLS_COMMON_CLIENT_CA = 2
    MTLS_SEPARATE_CLIENT_CA = 3


class ServerMode(Enum):
    HTTP = 0
    GRPC = 1
    BOTH = 2


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


## Tests #######################################################################


@pytest.mark.parametrize(
    "test_config",
    [
        (TlsMode.INSECURE, True),
        (TlsMode.TLS, True),
        (TlsMode.TLS, False),
        (TlsMode.MTLS_COMMON_CLIENT_CA, True),
        (TlsMode.MTLS_COMMON_CLIENT_CA, False),
        (TlsMode.MTLS_SEPARATE_CLIENT_CA, True),
        (TlsMode.MTLS_SEPARATE_CLIENT_CA, False),
    ],
)
@pytest.mark.parametrize("server_mode", ServerMode.__members__.values())
def test_health_probe(test_config: Tuple[TlsMode, bool], server_mode: ServerMode):
    """Test all of the different ways that the servers could be running"""
    tls_mode, inline = test_config

    # Get ports for both servers
    http_port = tls_test_tools.open_port()
    grpc_port = tls_test_tools.open_port()

    # Set up tls values if needed
    tls = tls_mode == TlsMode.TLS
    mtls = tls_mode in [TlsMode.MTLS_COMMON_CLIENT_CA, TlsMode.MTLS_SEPARATE_CLIENT_CA]
    with generate_tls_configs(
        port=http_port,
        tls=tls,
        mtls=mtls,
        inline=inline,
        separate_client_ca=tls_mode == TlsMode.MTLS_SEPARATE_CLIENT_CA,
    ) as config_overrides:
        with temp_probe_config(
            {
                "runtime": {
                    "grpc": {
                        "port": grpc_port,
                        "enabled": server_mode in [ServerMode.GRPC, ServerMode.BOTH],
                    },
                    "http": {
                        "enabled": server_mode in [ServerMode.HTTP, ServerMode.BOTH],
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
                    server_mode == ServerMode.GRPC
                )
                # If booting the HTTP server, do so
                with maybe_runtime_http_test_server(
                    http_port, tls_config_override=config_overrides
                ):
                    # Probe should always pass with both possible servers up
                    assert caikit_health_probe.health_probe()
