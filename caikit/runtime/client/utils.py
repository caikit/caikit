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
import json

# Third Party
from requests import Session
from requests.adapters import HTTPAdapter, Retry

# Third party
import grpc

# Local
from caikit.interfaces.common.data_model import ConnectionTlsInfo


def construct_grpc_channel(
    target: str,
    options: Optional[List[Tuple[str, str]]] = None,
    tls: Optional[ConnectionTlsInfo] = None,
    retries: Optional[int] = None,
    retry_options: Optional[Dict[str, str]] = None,
) -> grpc.Channel:
    """Helper function to construct a grpc Channel with the given TLS config

    Args:
        target (str): The target hostname
        options (Optional[List[Tuple[str, str]]], optional): List of tuples representing GRPC
            options. Defaults to None.
        tls (Optional[ConnectionTlsInfo], optional): The TLS information for this channel.
            Defaults to None.
        retries (Optional[int], optional): The max number of retries to attempt. Defaults to None.
        retry_options (Optional[Dict[str, str]], optional): Dictionary to override fields
            in the GRPC retry service config. Defaults to None.

    Returns:
        grpc.Channel: The constructed channel
    """
    # Add retry option if one was provided
    if retries and retries > 1:
        options.append(("grpc.enable_retries", 1))

        # Only add service_config if it wasn't already added to the GRPC option
        # this stops us from overriding an advanced config
        options_contain_service_config = False
        for option_name, _ in options:
            if option_name == "grpc.service_config":
                options_contain_service_config = True
                break

        if not options_contain_service_config:
            service_config = {
                "methodConfig": [
                    {
                        "name": [{}],
                        "retryPolicy": {
                            "maxAttempts": retries,
                            "initialBackoff": "0.1s",
                            "maxBackoff": "1s",
                            "backoffMultiplier": 2,
                            "retryableStatusCodes": [
                                "UNAVAILABLE",
                                "UNKNOWN",
                                "INTERNAL",
                            ],
                            **retry_options,
                        },
                    }
                ]
            }

            options.append(("grpc.service_config", json.dumps(service_config)))

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
    retries: Optional[int] = None,
    retry_options: Optional[Dict[str, str]] = None,
) -> Session:
    """Helper function to construct a requests Session object with the given TLS
    config

    Args:
        options (Optional[Dict[str, str]], optional): Dictionary of request kwargs to pass to
            session creation. Defaults to None.
        tls (Optional[ConnectionTlsInfo], optional): The TLS information for this session.
            Defaults to None.
        retries (Optional[int], optional): The max number of retries to attempt. Defaults to None.
        retry_options (Optional[Dict[str, str]], optional): Dictionary to override kwargs passed
            to the Retry object construction

    Returns:
        Session: _description_
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

    # Mount retry object if options were provided
    if retries:
        default_status_codes = list(Retry.RETRY_AFTER_STATUS_CODES) + [500, 502, 504]
        requests_retry = Retry(
            total=retries,
            allowed_methods=None,
            status_forcelist=default_status_codes,
            **(retry_options or {})
        )
        session.mount("http://", HTTPAdapter(max_retries=requests_retry))
        session.mount("https://", HTTPAdapter(max_retries=requests_retry))

    return session
