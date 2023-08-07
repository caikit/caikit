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
"""This module holds common utilities for managing timestamps in protobufs
"""
# Standard
import datetime

# Third Party
from google.protobuf import timestamp_pb2

# First Party
import alog

# Local
from ..exceptions import error_handler

log = alog.use_channel("TIMESTAMPS")
error = error_handler.get(log)

TIMESTAMP_PROTOBUF_NAME = "google.protobuf.Timestamp"


def datetime_to_proto(datetime_: datetime.datetime) -> timestamp_pb2.Timestamp:
    """Builds a google.protobuf.timestamp_pb2.Timestamp out of the provided datetime.datetime"""
    error.type_check("<COR48155462E>", datetime.datetime, datetime_=datetime_)
    return timestamp_pb2.Timestamp(
        seconds=int(datetime_.timestamp()), nanos=int(datetime_.microsecond * 1e3)
    )


def proto_to_datetime(time_pb2: timestamp_pb2.Timestamp) -> datetime.datetime:
    """Builds a datetime.datetime out of the provided google.protobuf.timestamp_pb2.Timestamp"""
    try:
        error.type_check("<COR48166462E>", timestamp_pb2.Timestamp, time_pb2=time_pb2)
    except TypeError as err:
        # Compatibility for some python 3.8 / protobuf 3.x setups
        if "Timestamp" not in str(type(time_pb2)):
            raise err

    return datetime.datetime.fromtimestamp(time_pb2.seconds + (time_pb2.nanos / 1e9))
