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

# First Party
import alog

# Local
from caikit.core.data_model import PACKAGE_COMMON, ProducerId, dataobject
from caikit.core.toolkit.errors import error_handler

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(
    package=PACKAGE_COMMON,
    schema={"producers": {"elements": ProducerId}},
)
class ProducerPriority:
    """An ordered list of ProducerId structures in descending order of priority.
    This is used when handling conflicts between multiple producers of the same
    data structure.
    """

    def __init__(self, producers):
        """Construct a new ProducerPriority

        Args:
            producers:  list(ProducerId)
        """
        error.type_check_all("<COR01353088E>", ProducerId, producers=producers)
        self.producers = producers
