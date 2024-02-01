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
Helper utils for GRPC and HTTP connections
"""
# Standard
from typing import Dict, List, Optional, Tuple

# Third Party
from requests import Session

# Third party
import grpc

# Local
from caikit.interfaces.common.data_model import ConnectionTlsInfo


def construct_grpc_channel(
    target: str,
    options: Optional[List[Tuple[str, str]]] = None,
    tls: Optional[ConnectionTlsInfo] = None,
) -> grpc.Channel:
    """Helper function to construct a grpc Channel with the given TLS config"""
    if tls and tls.enabled:
        grpc_credentials = grpc.ssl_channel_credentials(
            root_certificates=tls.ca_data,
            private_key=tls.key_data,
            certificate_chain=tls.cert_data,
        )
        return grpc.secure_channel(
            target, credentials=grpc_credentials, options=options
        )

    return grpc.insecure_channel(target, options=options)


def construct_requests_session(
    options: Optional[Dict[str, str]] = None,
    tls: Optional[ConnectionTlsInfo] = None,
    timeout: Optional[int] = None,
) -> Session:
    """Helper function to construct a requests Session object with the given TLS
    config
    """
    session = Session()
    session.headers["Content-type"] = "application/json"

    # Gather request SSL configuration
    if tls.enabled:
        # Configure the TLS CA settings
        if tls.insecure_verify:
            session.verify = False
        else:
            session.verify = tls.ca_file or True

        # Configure MTLS if its enabled
        if tls.mtls_enabled:
            session.cert = (
                tls.cert_file,
                tls.key_file,
            )

    # Update request options and timeout variables
    if options:
        session.params.update(options)

    if timeout:
        session.params["timeout"] = timeout

    return session
