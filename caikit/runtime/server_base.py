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
"""Base class with common functionality across all caikit servers"""

# Standard
from typing import Optional
import socket

# First Party
import aconfig
import alog

# Local
from caikit.config import get_config

log = alog.use_channel("SERVR-BASE")


class RuntimeServerBase:
    __doc__ = __doc__

    def __init__(self, base_port: int, tls_config_override: Optional[aconfig.Config]):
        self.config = get_config()
        self.port = (
            self._find_port(base_port)
            if self.config.runtime.find_available_ports
            else base_port
        )
        if self.port != base_port:
            log.warning(
                "Port %d was in use, had to find another! %d", base_port, self.port
            )
        print("in server base: ")
        print(tls_config_override)
        self.tls_config = (
            tls_config_override if tls_config_override else self.config.runtime.tls
        )

    @classmethod
    def _find_port(cls, start=8888, end=None, host="127.0.0.1"):
        """Function to find an available port in a given range
        Args:
            start: int
                Starting number for port search (inclusive)
                Default: 8888
            end: Optional[int]
                End number for port search (exclusive)
                Default: start + 1000
            host: str
                Host name or ip address to search on
                Default: localhost
        Returns:
            int
                Available port
        """
        end = start + 1000 if end is None else end
        if start < end:
            with socket.socket() as soc:
                # soc.connect_ex returns 0 if connection is successful and thus
                # indicating port is available
                if soc.connect_ex((host, start)) == 0:
                    # port is in use, thus connection to it is successful
                    return cls._find_port(start + 1, end, host)

                # port is open
                return start

    # Context manager impl
    def __enter__(self):
        self.start(blocking=False)
        return self

    def __exit__(self, type_, value, traceback):
        self.stop(0)
