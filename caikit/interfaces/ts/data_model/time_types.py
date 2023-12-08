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
The core data model objects for primitive time types
"""

# Standard
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import List, Tuple, Union
import json

# Third Party
import numpy as np

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber, OneofField
import alog

# Local
from ....core import DataObjectBase
from ....core.data_model import dataobject
from ....core.exceptions import error_handler
from .package import TS_PACKAGE

log = alog.use_channel("TSDM")
error = error_handler.get(log)


@dataobject(package=TS_PACKAGE)
class Seconds(DataObjectBase):
    """A nanosecond value that can be interpreted as either a datetime or a
    timedelta
    """

    seconds: Annotated[float, FieldNumber(1)]

    def as_datetime(self) -> datetime:
        """Return a python datetime object.
        The returned object will have timezone.utc set as its timezone info."""
        return datetime.fromtimestamp(self.seconds, tz=timezone.utc)

    def as_timedelta(self) -> timedelta:
        """Interpret these nanoseconds as a duration"""
        return timedelta(seconds=self.seconds)

    @classmethod
    def from_datetime(cls, time_point: datetime) -> "Seconds":
        """Create a Seconds from a datetime"""
        return cls(seconds=time_point.timestamp())

    @classmethod
    def from_timedelta(cls, time_delta: timedelta) -> "Seconds":
        """Create a Seconds from a timedelta"""
        return cls(seconds=time_delta.total_seconds())


@dataobject(package=TS_PACKAGE)
class TimePoint(DataObjectBase):
    """
    The core data model object for a TimePoint
    """

    time: Union[
        Annotated[int, OneofField("ts_int"), FieldNumber(1)],
        Annotated[float, OneofField("ts_float"), FieldNumber(2)],
        Annotated[Seconds, OneofField("ts_epoch"), FieldNumber(3)],
    ]


@dataobject(package=TS_PACKAGE)
class TimeDuration(DataObjectBase):
    """
    The core data model object for a TimeDuration
    """

    time: Union[
        Annotated[int, OneofField("dt_int"), FieldNumber(1)],
        Annotated[float, OneofField("dt_float"), FieldNumber(2)],
        Annotated[str, OneofField("dt_str"), FieldNumber(3)],
        Annotated[Seconds, OneofField("dt_sec"), FieldNumber(4)],
    ]


@dataobject(package=TS_PACKAGE)
class PeriodicTimeSequence(DataObjectBase):
    """A PeriodicTimeSequence represents an indefinite time sequence where ticks
    occur at a regular period
    """

    start_time: Annotated[TimePoint, FieldNumber(1)]
    period_length: Annotated[TimeDuration, FieldNumber(2)]


@dataobject(package=TS_PACKAGE)
class PointTimeSequence(DataObjectBase):
    """A PointTimeSequence represents a finite sequence of time points that may
    or may not be evenly distributed in time
    """

    points: Annotated[List[TimePoint], FieldNumber(1)]


@dataobject(package=TS_PACKAGE)
class Vector(DataObjectBase):
    """A vector represents a finite sequence of doubles"""

    data: Annotated[List[float], FieldNumber(1)]


@dataobject(package=TS_PACKAGE)
class ValueSequence(DataObjectBase):
    """A ValueSequence is a finite list of contiguous values, each representing
    the value of a given attribute for a specific observation within a
    TimeSeries
    """

    @dataobject(package=TS_PACKAGE)
    class IntValueSequence(DataObjectBase):
        """Nested value sequence of integers"""

        values: Annotated[List[int], FieldNumber(1)]

    @dataobject(package=TS_PACKAGE)
    class FloatValueSequence(DataObjectBase):
        """Nested value sequence of floats"""

        values: Annotated[List[float], FieldNumber(1)]

    @dataobject(package=TS_PACKAGE)
    class StrValueSequence(DataObjectBase):
        """Nested value sequence of strings"""

        values: Annotated[List[str], FieldNumber(1)]

    @dataobject(package=TS_PACKAGE)
    class VectorValueSequence(DataObjectBase):
        """Nested value sequence of vectors"""

        values: Annotated[List[Vector], FieldNumber(1)]

        def __post_init__(self):
            error.type_check("<COR61333730E>", list, values=self.values)
            error.type_check_all(
                "<COR61333731E>", list, np.ndarray, Vector, values=self.values
            )

        def _convert_np_to_list(self, v):
            return v.tolist()

        def to_dict(self):
            result = []
            for v in self.values:
                v_in = self._convert_np_to_list(v) if isinstance(v, np.ndarray) else v
                result.append({"data": v_in if isinstance(v_in, list) else v.data})
            return {"values": result}

        def fill_proto(self, proto):
            subproto = proto.values
            subproto.extend(
                [
                    v.to_proto()
                    if isinstance(v, Vector)
                    else Vector(
                        v if isinstance(v, list) else self._convert_np_to_list(v)
                    ).to_proto()
                    for v in self.values
                ]
            )
            return proto

        @classmethod
        def from_proto(cls, proto):
            return cls(**{"values": [list(v.data) for v in proto.values]})

    # todo we can have a constuct for sequences that require serialization
    @dataobject(package=TS_PACKAGE)
    class TimePointSequence(DataObjectBase):
        """Nested value sequence of TimePoints"""

        values: Annotated[List[str], FieldNumber(1)]

    # todo we can have a construct for sequences that require serialization
    @dataobject(package=TS_PACKAGE)
    class AnyValueSequence(DataObjectBase):
        """Nested value sequence of Any objects"""

        values: Annotated[List[str], FieldNumber(1)]

        @classmethod
        @lru_cache(maxsize=None)
        def decode_values(cls, values: Tuple[str]):
            """Cached class method to enable caching of decoded representations"""
            return [json.loads(v) for v in values]

        def to_dict(self):
            return {"values": self.__class__.decode_values(tuple(self.values))}

        def fill_proto(self, proto):
            subproto = proto.values
            subproto.extend(self.__class__.decode_values(tuple(self.values)))
            return proto

        @classmethod
        def from_proto(cls, proto):
            return cls(**{"values": [json.dumps(v) for v in proto.values]})

    sequence: Union[
        Annotated[IntValueSequence, OneofField("val_int"), FieldNumber(1)],
        Annotated[FloatValueSequence, OneofField("val_float"), FieldNumber(2)],
        Annotated[StrValueSequence, OneofField("val_str"), FieldNumber(3)],
        Annotated[TimePointSequence, OneofField("val_timepoint"), FieldNumber(4)],
        Annotated[AnyValueSequence, OneofField("val_any"), FieldNumber(5)],
        Annotated[VectorValueSequence, OneofField("val_vector"), FieldNumber(6)],
    ]
