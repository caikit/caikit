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

# Standard
from typing import List

# First Party
import alog

# Local
from caikit.core.data_model import (
    PACKAGE_COMMON,
    DataObjectBase,
    ProducerId,
    dataobject,
)
from caikit.core.toolkit.errors import error_handler

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(PACKAGE_COMMON)
class ProducerPriority(DataObjectBase):
    """An ordered list of ProducerId structures in descending order of priority.
    This is used when handling conflicts between multiple producers of the same
    data structure.
    """

    producers: List[ProducerId]
