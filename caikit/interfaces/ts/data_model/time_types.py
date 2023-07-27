"""
The core data model objects for primitive time types
"""

# Standard
from datetime import datetime, timedelta, timezone
from typing import List, Union
import json

try:
    # Standard
    from typing import Annotated
except ImportError:  # pragma: no cover
    # Third Party
    from typing_extensions import Annotated


# Third Party
import numpy as np

# First Party
from py_to_proto.dataclass_to_proto import FieldNumber, OneofField
import alog

# Local
from .package import TS_PACKAGE
from caikit.core import DataObjectBase
from caikit.core.data_model import dataobject

log = alog.use_channel("TSDM")


@dataobject(package=TS_PACKAGE)
class Seconds(DataObjectBase):
    """A nanosecond value that can be interpreted as either a datetime or a
    timedelta
    """

    seconds: float

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

    def to_dict(self) -> dict:
        return {"seconds": self.seconds}


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

    start_time: TimePoint
    period_length: TimeDuration


@dataobject(package=TS_PACKAGE)
class PointTimeSequence(DataObjectBase):
    """A PointTimeSequence represents a finite sequence of time points that may
    or may not be evenly distributed in time
    """

    points: List[TimePoint]


@dataobject(package=TS_PACKAGE)
class Vector(DataObjectBase):
    """A vector represents a finite sequence of doubles"""

    data: List[float]


@dataobject(package=TS_PACKAGE)
class ValueSequence(DataObjectBase):
    """A ValueSequence is a finite list of contiguous values, each representing
    the value of a given attribute for a specific observation within a
    TimeSeries
    """

    @dataobject(package=TS_PACKAGE)
    class IntValueSequence(DataObjectBase):
        """Nested value sequence of integers"""

        values: List[int]

    @dataobject(package=TS_PACKAGE)
    class FloatValueSequence(DataObjectBase):
        """Nested value sequence of floats"""

        values: List[float]

    @dataobject(package=TS_PACKAGE)
    class StrValueSequence(DataObjectBase):
        """Nested value sequence of strings"""

        values: List[str]

    @dataobject(package=TS_PACKAGE)
    class VectorValueSequence(DataObjectBase):
        """Nested value sequence of vectors"""

        values: List[Vector]

        def _convert_np_to_list(self, v):
            v = v.tolist()
            return v

        def to_dict(self):
            result = []
            for v in self.values:
                if isinstance(v, np.ndarray):
                    v_in = self._convert_np_to_list(v)
                # we don't create these at the application leve
                # It should emerge only from to/from_proto invocations
                # elif isinstance(v, Vector):
                #    v_in = v.data
                else:
                    v_in = v

                result.append({"data": v_in if isinstance(v_in, list) else v.data})
            return {"values": result}

        def fill_proto(self, proto):
            subproto = getattr(proto, "values")
            subproto.extend(
                [
                    Vector.from_json(
                        {
                            "data": v
                            if isinstance(v, list)
                            else self._convert_np_to_list(v)
                        }
                    ).to_proto()
                    for v in self.values
                ]
            )

        @classmethod
        def from_proto(cls, proto):
            return cls(**{"values": [list(v.data) for v in proto.values]})

    # todo we can have a constuct for sequences that require serialization
    @dataobject(package=TS_PACKAGE)
    class TimePointSequence(DataObjectBase):
        """Nested value sequence of TimePoints"""

        values: List[str]

        def to_dict(self):
            result = []
            for v in self.values:
                result.append(v)
            return {"values": result}

        def fill_proto(self, proto):
            subproto = getattr(proto, "values")
            subproto.extend([v for v in self.values])

        @classmethod
        def from_proto(cls, proto):
            return cls(**{"values": [str(v) for v in proto.values]})

    # todo we can have a construct for sequences that require serialization
    @dataobject(package=TS_PACKAGE)
    class AnyValueSequence(DataObjectBase):
        """Nested value sequence of Any objects"""

        values: List[str]

        def to_dict(self):
            result = []
            for v in self.values:
                json_v = json.loads(v)
                result.append(json_v)
            return {"values": result}

        def fill_proto(self, proto):
            subproto = getattr(proto, "values")
            subproto.extend([json.loads(v) for v in self.values])

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
