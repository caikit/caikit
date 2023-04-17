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

"""Tests for producer"""

# Local
from caikit.core.data_model.producer import ProducerId


def test_add_producer_ids():
    prod_id_1 = ProducerId("foo", "1.2.3")
    prod_id_2 = ProducerId("bar", "1.2.3")
    producer = prod_id_1 + prod_id_2
    assert producer.name == "foo & bar"


def test_proto_roundtrip():
    prod_id = ProducerId("foo", "1.2.3")
    prod_id_roundtrip = ProducerId.from_proto(prod_id.to_proto())
    assert prod_id.name == prod_id_roundtrip.name
    assert prod_id.version == prod_id_roundtrip.version
