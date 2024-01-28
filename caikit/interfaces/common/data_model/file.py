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
This file contains interfaces required to handle File data
"""

# Standard
from dataclasses import dataclass
from typing import Optional

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber

# Local
from caikit.core.data_model import PACKAGE_COMMON, DataObjectBase, dataobject


@dataobject(PACKAGE_COMMON)
@dataclass
class File(DataObjectBase):
    data: Annotated[bytes, FieldNumber(1)]
    filename: Annotated[Optional[str], FieldNumber(2)]
    type: Annotated[Optional[str], FieldNumber(3)]
