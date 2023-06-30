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
"""Tests for producer priority"""


# Local
from caikit.core.data_model import ProducerId
from caikit.interfaces.common.data_model.producer import ProducerPriority

## Setup #####################################################################
PROD_ID_1 = ProducerId("foo", "1.2.3")
PROD_ID_2 = ProducerId("bar", "1.2.3")

## Tests #####################################################################


def test_proto_roundtrip():
    producer_priority = ProducerPriority(producers=[PROD_ID_1, PROD_ID_2])
    producer_priority_roundtrip = ProducerPriority.from_proto(
        producer_priority.to_proto()
    )
    assert producer_priority_roundtrip.producers[0] == PROD_ID_1
    assert producer_priority_roundtrip.producers[1] == PROD_ID_2


def test_json_roundtrip():
    producer_priority = ProducerPriority(producers=[PROD_ID_1, PROD_ID_2])
    producer_priority_roundtrip = ProducerPriority.from_json(
        producer_priority.to_json()
    )
    assert producer_priority_roundtrip.producers[0] == PROD_ID_1
    assert producer_priority_roundtrip.producers[1] == PROD_ID_2
