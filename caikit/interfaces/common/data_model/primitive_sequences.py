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
This file contains interfaces required for unions of lists
"""

# Standard
from dataclasses import dataclass
from typing import Any, List

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber

# Local
from caikit.core.data_model import PACKAGE_COMMON, DataObjectBase, dataobject


class Sequence(DataObjectBase):
    """Base class for all Sequences to enable type checking
    e.g. isinstance(<>, Sequence)"""

    values: List[Any]


@dataobject(PACKAGE_COMMON)
@dataclass
class IntSequence(Sequence):
    values: Annotated[List[int], FieldNumber(1)]


@dataobject(PACKAGE_COMMON)
@dataclass
class FloatSequence(Sequence):
    values: Annotated[List[float], FieldNumber(1)]


@dataobject(PACKAGE_COMMON)
@dataclass
class StrSequence(Sequence):
    values: Annotated[List[str], FieldNumber(1)]


@dataobject(PACKAGE_COMMON)
@dataclass
class BoolSequence(Sequence):
    values: Annotated[List[bool], FieldNumber(1)]
