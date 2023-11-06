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
This module implements a common health probe for all running runtime servers.
"""
# Standard
from contextlib import contextmanager
from typing import Optional, Tuple
import importlib.util
import os
import sys
import tempfile
import warnings

# First Party
import alog

# We play some tricks to import caikit's config here and avoid importing all of
# caikit itself. The reason for this is that importing caikit can be very costly
# relative to the actual cost of probing the servers and since this is intended
# to stand alone as an executable, that import time cost is added to every call.
caikit_spec = importlib.util.find_spec("caikit")
sys.path = [os.path.dirname(caikit_spec.origin)] + sys.path
# Third Party
from config import get_config

sys.path = sys.path[1:]


log = alog.use_channel("PROBE")


@alog.timed_function(log.debug)
def health_probe() -> bool:
    """Run a health probe against all running runtime servers.

    This function is intended to be run from an environment where the config is
    identical to the config that the server is running such as from inside a
    kubernetes pod where the server is also running.

    Returns:
        healthy (bool): True if all servers are healthy, False otherwise
    """

    # Get TLS key/cert files if possible
    config = get_config()
    tls_key = config.runtime.tls.server.key
    tls_cert = config.runtime.tls.server.cert
    client_ca = config.runtime.tls.client.cert
    http_healthy, grpc_healthy = None, None

    if config.runtime.http.enabled:
        log.debug("Checking HTTP server health")
        http_healthy = _http_health_probe(
            config.runtime.http.port, tls_key, tls_cert, client_ca
        )

    if config.runtime.grpc.enabled:
        log.debug("Checking gRPC server health")
        grpc_healthy = _grpc_health_probe(
            config.runtime.grpc.port, tls_key, tls_cert, client_ca
        )

    if False in [http_healthy, grpc_healthy]:
        log.info(
            "<RUN64273066I>",
            "Server not healthy. HTTP: %s, gRPC: %s",
            http_healthy,
            grpc_healthy,
        )
        return False
    return True


## Implementation ##############################################################


def _http_health_probe(
    port: int,
    tls_key: Optional[str],
    tls_cert: Optional[str],
    client_ca: Optional[str],
) -> bool:
    """Probe the http server

    The implementation of this utility is a bit tricky because mTLS makes this
    quite challenging. For insecure or TLS servers, we expect a valid healthy
    response, but for mTLS servers, we may not have a valid key/cert pair that
    the client can present to the server that is signed by the expected CA if
    the trusted client CA does not match the one that signed the server's
    key/cert pair.

    The workaround for this is to detect SSLError and consider that to be a
    passing health check. If the server is healthy enough to _reject_ bad SSL
    requests, it's healthy enough to server good ones!

    Args:
        port (int): The port that the HTTP server is serving on
        tls_key (Optional[str]): Body or path to the TLS key file if TLS/mTLS
            enabled
        tls_cert (Optional[str]): Body or path to the TLS cert file if TLS/mTLS
            enabled
        client_ca (Optional[str]): The client ca cert that the server is using
            for mutual client auth

    Returns:
        healthy (bool): True if all servers are healthy, False otherwise
    """
    # NOTE: Local imports for optional dependency
    with alog.ContextTimer(log.debug2, "Done with local grpc imports: "):

        # Third Party
        import requests  # pylint: disable=import-outside-toplevel

    # Requests requires that the TLS information be in files
    with _tls_files(tls_key, tls_cert) as tls_files:
        key_file, cert_file = tls_files
        if key_file and cert_file:
            protocol = "https"
            kwargs = {"verify": False}
            if client_ca:
                log.debug("Probing mTLS HTTP Server")
                kwargs["cert"] = (key_file, cert_file)
            else:
                log.debug("Probing TLS HTTP Server")
        else:
            log.debug("Probing INSECURE HTTP Server")
            protocol = "http"
            kwargs = {}

        try:
            # Suppress insecure connection warnings since we disable server
            # verification. This is ok since this probe will be run against
            # localhost in a pod where the server is _known_ to be authentic.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", module="urllib3")
                session = requests.Session()
                retries = requests.adapters.Retry(total=0)
                session.mount(
                    f"{protocol}://", requests.adapters.HTTPAdapter(max_retries=retries)
                )
                # NOTE: Not using the constant to avoid big imports
                resp = session.get(
                    f"{protocol}://localhost:{port}/health",
                    timeout=0.01,
                    **kwargs,
                )
            resp.raise_for_status()
            return True
        except requests.exceptions.SSLError as err:
            log.debug("Got SSLError indicating a healthy SSL server! %s", err)
            log.debug2(err, exc_info=True)
            return True
        except Exception as err:  # pylint: disable=broad-exception-caught
            log.debug2("Caught unexpected error: %s", err, exc_info=True)
        return False


def _grpc_health_probe(
    port: int,
    tls_key: Optional[str],
    tls_cert: Optional[str],
    client_ca: Optional[str],
) -> bool:
    """Probe the grpc server

    Since the gRPC server trusts its own cert for client verification, we can
    make a valid health probe against the running server regardless of (m)TLS
    config.

    Args:
        port (int): The port that the gRPC server is serving on
        tls_key (Optional[str]): Body or path to the TLS key file if TLS/mTLS
            enabled
        tls_cert (Optional[str]): Body or path to the TLS cert file if TLS/mTLS
            enabled
        client_ca (Optional[str]): The client ca cert that the server is using
            for mutual client auth

    Returns:
        healthy (bool): True if all servers are healthy, False otherwise
    """
    # NOTE: Local imports for optional dependency
    with alog.ContextTimer(log.debug2, "Done with local grpc imports: "):

        # Third Party
        from grpc_health.v1 import (  # pylint: disable=import-outside-toplevel
            health_pb2,
            health_pb2_grpc,
        )
        import grpc  # pylint: disable=import-outside-toplevel

    # Server hostname to use unless using socket mode
    hostname = f"localhost:{port}"
    socket_file = get_config().runtime.grpc.unix_socket_path

    # If available, use a unix socket
    if socket_file and os.path.exists(os.path.dirname(socket_file)):
        socket_address = f"unix://{socket_file}"
        log.debug("Probing gRPC server over unix socket: %s", socket_file)
        channel = grpc.insecure_channel(socket_address)

    elif tls_key and tls_cert:
        tls_server_key = bytes(_load_secret(tls_key), "utf-8")
        tls_server_cert = bytes(_load_secret(tls_cert), "utf-8")
        if client_ca:
            log.debug("Probing mTLS gRPC server")
            credentials = grpc.ssl_channel_credentials(
                root_certificates=tls_server_cert,
                private_key=tls_server_key,
                certificate_chain=tls_server_cert,
            )
        else:
            log.debug("Probing TLS gRPC server")
            credentials = grpc.ssl_channel_credentials(
                root_certificates=tls_server_cert,
            )

        # NOTE: If the server's certificate does not have 'localhost' in it,
        #   this will cause certificate validation errors and fail. The original
        #   workaround for this was to parse the cert's SANs and use hostname
        #   overrides, but that requires a full cryptographic PEM parser which
        #   is a security-sensitive dependency to pull that we want to avoid.
        #   Instead, the workaround is to use the unix socket server option
        #   above.
        channel = grpc.secure_channel(hostname, credentials=credentials)
    else:
        log.debug("Probing INSECURE gRPC server")
        channel = grpc.insecure_channel(hostname)

    client = health_pb2_grpc.HealthStub(channel)
    try:
        client.Check(health_pb2.HealthCheckRequest())
        return True
    except Exception as err:  # pylint: disable=broad-exception-caught
        log.debug2("Caught unexpected error: %s", err, exc_info=True)
        return False


@contextmanager
def _tls_files(
    tls_key: Optional[str],
    tls_cert: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Get files for the TLS key/cert if given"""
    if not tls_key or not tls_cert:
        yield None, None
        return
    valid_file_vals = [
        ((os.path.exists(fname) and fname) or None) for fname in [tls_key, tls_cert]
    ]
    if all(valid_file_vals):
        yield tls_key, tls_cert
        return
    with tempfile.TemporaryDirectory() as workdir:
        key_file, cert_file = valid_file_vals
        if not key_file:
            key_file = os.path.join(workdir, "tls.key")
            with open(key_file, "w", encoding="utf-8") as handle:
                handle.write(tls_key)
        if not cert_file:
            cert_file = os.path.join(workdir, "tls.cert")
            with open(cert_file, "w", encoding="utf-8") as handle:
                handle.write(tls_cert)
        yield key_file, cert_file
        return


def _load_secret(secret: str) -> str:
    """NOTE: Copied from grpc_server to avoid costly imports"""
    if os.path.exists(secret):
        with open(secret, "r", encoding="utf-8") as secret_file:
            return secret_file.read()
    return secret


## Main ########################################################################
def main():
    caikit_config = get_config()
    alog.configure(
        default_level=caikit_config.log.level,
        filters=caikit_config.log.filters,
        thread_id=caikit_config.log.thread_id,
        formatter=caikit_config.log.formatter,
    )
    if not health_probe():
        sys.exit(1)


if __name__ == "__main__":
    main()
