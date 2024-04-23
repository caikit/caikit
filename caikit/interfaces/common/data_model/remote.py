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
This file contains interfaces required to connect to Remote servers
"""

# Standard
from dataclasses import field
from http.client import HTTP_PORT, HTTPS_PORT
from pathlib import Path
from typing import Optional

# First Party
from py_to_proto.dataclass_to_proto import Dict
import alog

# Local
from caikit.core.data_model import PACKAGE_COMMON, DataObjectBase, dataobject
from caikit.core.exceptions import error_handler

log = alog.use_channel("CNNCTDM")
error = error_handler.get(log)


@dataobject(PACKAGE_COMMON)
class ConnectionTlsInfo(DataObjectBase):
    """Helper dataclass to store information regarding TLS information."""

    # If TLS is enabled
    enabled: bool = False

    # Whether to verify server CA bundle
    insecure_verify: bool = False

    # TLS Key information
    ca_file: Optional[str]
    cert_file: Optional[str]
    key_file: Optional[str]

    @property
    def mtls_enabled(self) -> bool:
        """Helper property to identify if mtls is enabled"""
        return self.cert_file and self.key_file

    # Don't use cached_property as DataBase does not contain a __dict__ object
    # This also required provided private_slots to DataBase
    _private_slots = ("_ca_data", "_cert_data", "_key_data")

    @property
    def ca_data(self) -> Optional[bytes]:
        if not self._ca_data and self.ca_file and Path(self.ca_file).exists():
            self._ca_data = Path(self.ca_file).read_bytes()
        return self._ca_data

    @property
    def key_data(self) -> Optional[bytes]:
        if not self._key_data and self.key_file and Path(self.key_file).exists():
            self._key_data = Path(self.key_file).read_bytes()
        return self._key_data

    @property
    def cert_data(self) -> Optional[bytes]:
        if not self._cert_data and self.cert_file and Path(self.cert_file).exists():
            self._cert_data = Path(self.cert_file).read_bytes()
        return self._cert_data

    def __post_init__(self):
        """Post init function to verify field types and arguments"""
        error.type_check(
            "<COR734221567E>",
            str,
            bytes,
            allow_none=True,
            tls_ca=self.ca_file,
            tls_cert=self.cert_file,
            key_file=self.key_file,
        )

        error.type_check(
            "COR74322567E",
            bool,
            tls_enabled=self.enabled,
            insecure_verify=self.insecure_verify,
        )

        # Initialize cached properties
        self._ca_data = None
        self._cert_data = None
        self._key_data = None

        # Read file data if it exists
        if self.enabled:
            self.verify_ssl_data()

    def verify_ssl_data(self):
        """Helper function to verify all TLS data was read correctly.

        Raises:
            FileNotFoundError: If any of the tls files were provided but could not be found
        """
        if self.ca_file and not self.ca_data:
            raise FileNotFoundError(f"Unable to find TLS CA File {self.ca_file}")
        if self.key_file and not self.key_data:
            raise FileNotFoundError(f"Unable to find TLS Key File {self.key_file}")
        if self.cert_file and not self.cert_data:
            raise FileNotFoundError(f"Unable to find TLS Cert File {self.cert_file}")

        # Logical XOR to ensure if one is provided so is the other
        if bool(self.cert_file) != bool(self.key_file):
            raise ValueError(
                "Invalid TLS values. Both cert and key must be provided:"
                f"{self.cert_file=}, {self.key_file=}"
            )


@dataobject(PACKAGE_COMMON)
class ConnectionInfo(DataObjectBase):
    """DataClass to store information regarding an external connection. This includes the hostname,
    port, tls, and timeout settings"""

    # Generic Host settings
    hostname: str
    port: Optional[int] = None

    # TLS Settings
    tls: Optional[ConnectionTlsInfo] = field(default_factory=ConnectionTlsInfo)

    # Connection timeout settings (in seconds)
    timeout: Optional[int] = 60

    # Any extra options for the connection
    options: Optional[Dict[str, str]] = field(default_factory=dict)

    def __post_init__(self):
        """Post init function to verify field types and set defaults"""

        # If tls is attribute dict then manually convert it to tls
        if isinstance(self.tls, dict):
            self.tls = ConnectionTlsInfo(**self.tls)

        # Set default port. Utilize the standard HTTP ports as the majority of protocols
        # use http under the hood like grpc and s3
        if not self.port:
            self.port = HTTPS_PORT if self.tls.enabled else HTTP_PORT

        # Type check all arguments
        error.type_check(
            "<COR734221567E>",
            str,
            hostname=self.hostname,
        )

        error.type_check("<COR734224567E>", int, port=self.port, timeout=self.timeout)

        if self.options:
            error.type_check("<COR734424567E>", str, int, float, **self.options)
