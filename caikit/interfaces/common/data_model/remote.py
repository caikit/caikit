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
from py_to_proto.dataclass_to_proto import Annotated, Dict, FieldNumber
import alog

# Local
from caikit.core.data_model import PACKAGE_COMMON, DataObjectBase, dataobject
from caikit.core.exceptions import error_handler

log = alog.use_channel("CNNCTDM")
error = error_handler.get(log)


@dataobject(PACKAGE_COMMON)
class ConnectionTlsInfo(DataObjectBase):
    """Helper dataclass to store information regarding TLS information."""

    enabled: Annotated[bool, FieldNumber(1)] = False
    ca_file: Annotated[Optional[str], FieldNumber(2)]
    cert_file: Annotated[Optional[str], FieldNumber(3)]
    key_file: Annotated[Optional[str], FieldNumber(4)]
    insecure_verify: Annotated[bool, FieldNumber(5)] = False

    # Helper variables to store the read data from TLS files
    ca_file_data: Optional[bytes] = None
    cert_file_data: Optional[bytes] = None
    key_file_data: Optional[bytes] = None

    def __post_init__(self):
        """Post init function to verify field types and arguments"""
        error.type_check(
            "<COR734221567E>",
            str,
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

        # Read file data if it exists
        if self.ca_file and Path(self.ca_file).exists():
            self.ca_file_data = Path(self.ca_file).read_bytes()

        if self.cert_file and Path(self.cert_file).exists():
            self.cert_file_data = Path(self.cert_file).read_bytes()

        if self.key_file and Path(self.key_file).exists():
            self.key_file_data = Path(self.key_file).read_bytes()

        if self.enabled:
            self.verify_ssl_data()

    def verify_ssl_data(self):
        """Helper function to verify all TLS data was read correctly.

        Raises:
            FileNotFoundError: If any of the tls files were provided but could not be found
        """
        if self.ca_file and not self.ca_file_data:
            raise FileNotFoundError(f"Unable to find TLS CA File {self.ca_file}")
        if self.key_file and not self.key_file_data:
            raise FileNotFoundError(f"Unable to find TLS Key File {self.key_file}")
        if self.cert_file and not self.cert_file_data:
            raise FileNotFoundError(f"Unable to find TLS Cert File {self.cert_file}")


@dataobject(PACKAGE_COMMON)
class ConnectionInfo(DataObjectBase):
    """DataClass to store information regarding an external connection. This includes the hostname,
    port, tls, and timeout settings"""

    # Generic Host settings
    hostname: Annotated[str, FieldNumber(1)]
    port: Annotated[Optional[int], FieldNumber(2)] = None

    # TLS Settings
    tls: Annotated[Optional[ConnectionTlsInfo], FieldNumber(3)] = field(
        default_factory=ConnectionTlsInfo
    )

    # Connection timeout settings
    timeout: Annotated[Optional[int], FieldNumber(4)] = 60

    # Any extra options for the connection
    options: Annotated[Optional[Dict[str, str]], FieldNumber(5)] = field(
        default_factory=dict
    )

    def __post_init__(self):
        """Post init function to verify field types and set defaults"""

        # Set default port
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
            error.type_check("<COR734424567E>", str, **self.options)
