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
Common data model objects used to identify the producer of a given output
"""
# Standard
from dataclasses import dataclass

# Local
from .dataobject import CAIKIT_DATA_MODEL, dataobject

PACKAGE_COMMON = f"{CAIKIT_DATA_MODEL}.common"


@dataobject(PACKAGE_COMMON)
@dataclass
class ProducerId:
    """Information about a data structure and the module that produced it."""

    name: str
    version: str

    def __add__(self, other):
        """Add two producer ids."""
        return ProducerId(name=" & ".join([self.name, other.name]), version="0.0.0")

    @classmethod
    def from_proto(cls, proto):
        """Overloaded implementation for efficiency vs base introspection"""
        return cls(proto.name, proto.version)

    def fill_proto(self, proto):
        """Overloaded implementation for efficiency vs base introspection"""
        proto.name = self.name
        proto.version = self.version
        return proto
