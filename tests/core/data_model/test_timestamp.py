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
Tests for conversion between python datetime and protobuf Timestamp
"""
# Standard
from typing import List
import datetime

# Third Party
import pytest

# Local
from caikit.core import DataObjectBase, dataobject
from caikit.core.data_model import timestamp
from caikit.interfaces.common.data_model.stream_sources import S3Path
from tests.core.data_model.test_dataobject import auto_reset_global_protobuf_registry


def test_timestamp_to_proto_to_timestamp():
    datetime_ = datetime.datetime.now()
    proto = timestamp.datetime_to_proto(datetime_)
    new_datetime = timestamp.proto_to_datetime(proto)

    assert new_datetime == datetime_


def test_timestamp_to_proto_invalid_type():
    """Make sure that a TypeError is raised if a bad type is encountered"""
    with pytest.raises(TypeError):
        timestamp.datetime_to_proto({"this is not": "a timestamp"})


def test_proto_to_timestamp_invalid_type():
    """Make sure that a TypeError is raised if a bad type is encountered"""
    with pytest.raises(TypeError):
        non_timestamp_proto = S3Path(path="/not/a/timestamp.either")
        timestamp.proto_to_datetime(non_timestamp_proto)


def test_timestamp_roundtrip_serialization():
    @dataobject(package="test")
    class FooTimestamps(DataObjectBase):
        time: datetime.datetime
        times: List[datetime.datetime]

    time_container = FooTimestamps(
        time=datetime.datetime.now(),
        times=[datetime.datetime.now(), datetime.datetime.fromtimestamp(1234567)],
    )

    time_proto = time_container.to_proto()
    time_proto_roundtrip = FooTimestamps.from_proto(time_proto)
    assert time_proto_roundtrip == time_container

    time_json = time_container.to_json()
    time_json_roundtrip = FooTimestamps.from_json(time_json)
    assert time_json_roundtrip == time_container
