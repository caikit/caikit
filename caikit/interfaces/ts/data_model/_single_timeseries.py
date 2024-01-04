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
The core data model object for a TimeSeries
"""
# Standard
from datetime import timedelta
from typing import Iterable, List, Optional, Tuple, Union
import json

# Third Party
import dateutil.parser
import numpy as np
import pandas as pd

# First Party
from py_to_proto.dataclass_to_proto import (  # Annotated imported from here for compatibility
    Annotated,
    FieldNumber,
    OneofField,
)
import alog

# Local
from ....core import DataObjectBase
from ....core.data_model import dataobject
from ....core.exceptions import error_handler
from ..data_model.backends.util import strip_periodic
from .backends.base import TimeSeriesBackendBase
from .backends.pandas_backends import PandasTimeSeriesBackend
from .package import TS_PACKAGE
from .time_types import PeriodicTimeSequence, PointTimeSequence, Seconds, ValueSequence
from .toolkit.optional_dependencies import HAVE_PYSPARK, pyspark
from .toolkit.sparkconf import sparkconf_local

log = alog.use_channel("TSDM")
error = error_handler.get(log)

## TimeSeries ##################################################################


@dataobject(package=TS_PACKAGE)
class SingleTimeSeries(DataObjectBase):
    """The TimeSeries object is the central data container for the library.
    At present it wraps either a pandas.DataFrame, or pyspark.sql.DataFrame to bind
    into the caikit data model.
    """

    @dataobject(package=TS_PACKAGE)
    class StringIDSequence(DataObjectBase):
        """Nested value sequence of strings"""

        values: Annotated[List[str], FieldNumber(1)]

    @dataobject(package=TS_PACKAGE)
    class IntIDSequence(DataObjectBase):
        """Nested value sequence of ints"""

        values: Annotated[List[int], FieldNumber(1)]

    time_sequence: Union[
        Annotated[PeriodicTimeSequence, OneofField("time_period"), FieldNumber(10)],
        Annotated[PointTimeSequence, OneofField("time_points"), FieldNumber(20)],
    ]
    values: Annotated[List[ValueSequence], FieldNumber(1)]
    timestamp_label: Annotated[str, FieldNumber(2)]
    value_labels: Annotated[List[str], FieldNumber(3)]
    ids: Union[
        Annotated[IntIDSequence, OneofField("id_int"), FieldNumber(30)],
        Annotated[StringIDSequence, OneofField("id_str"), FieldNumber(40)],
    ]

    _DEFAULT_TS_COL = "timestamp"

    # TODO: We need to clean up the init semantics
    def __init__(self, *args, **kwargs):
        """Constructing a TimeSeries directly always delegates to the pandas
        backend
        """
        # this is called from MultiTimeSeries
        if backend := kwargs.get("_backend", None):
            self._backend = backend
        elif "values" in kwargs:
            self._ids = None
            if "id_int" in kwargs:
                self._which_oneof_ids = "id_int"
                self._ids = kwargs["id_int"]
            if "id_str" in kwargs:
                self._which_oneof_ids = "id_str"
                self._ids = kwargs["id_str"]
            if "time_period" in kwargs:
                self._which_oneof_time_sequence = "time_period"
                self._time_sequence = kwargs["time_period"]
            if "time_points" in kwargs:
                self._which_oneof_time_sequence = "time_points"
                self._time_sequence = kwargs["time_points"]

            for k, v in kwargs.items():
                setattr(self, k, v)

        else:
            error.value_check(
                "<COR81128386E>",
                len(args) != 0,
                "must have at least the data argument",
                args,
            )
            data_arg = args[0]

            if isinstance(data_arg, pd.DataFrame):
                self._backend = PandasTimeSeriesBackend(*args, **kwargs)
            elif HAVE_PYSPARK and isinstance(data_arg, pyspark.sql.DataFrame):
                # Local
                # pylint: disable=import-outside-toplevel
                from .backends._spark_backends import SparkTimeSeriesBackend

                self._backend = SparkTimeSeriesBackend(*args, **kwargs)
            else:
                raise NotImplementedError("not implemented yet")

    def _get_pd_df(self) -> Tuple[pd.DataFrame, str, Iterable[str]]:
        """Convert the data to a pandas DataFrame, efficiently if possible"""

        # If there is a backend that knows how to do the conversion, use that
        backend = getattr(self, "_backend", None)
        if backend is not None and isinstance(backend, TimeSeriesBackendBase):
            log.debug("Using backend pandas conversion")
            return backend.as_pandas()

        # If not, convert the slow way from the proto representation
        df_kwargs = {}

        # Since all fields are optional, we need to ensure that the
        # time_sequence oneof has been set and that there are values
        error.value_check(
            "<COR07953363E>",
            self.time_sequence is not None,
            "Cannot create pandas data frame without a time sequence",
        )
        error.value_check(
            "<COR98388947E>",
            self.values is not None,
            "Cannot create pandas data frame without values",
        )

        # Determine the number of rows we'll expect
        col_lens = {len(col.sequence.values) for col in self.values}
        error.value_check(
            "<COR24439736E>",
            len(col_lens) == 1,
            "Not all columns have matching lengths",
        )
        num_rows = list(col_lens)[0]
        log.debug("Num rows: %d", num_rows)

        # If the time index is stored periodically, this can be represented as a
        # periodic index in pandas iff the start time and period are grounded in
        # real datetime space. If they are purely numerical, they can be
        # converted to a set of point values. The only invalid combination is a
        # numeric start time and a timedelta duration.
        #
        # (datetime, numeric) -> period w/ numeric seconds
        # (datetime, str) -> period w/ string freq
        # (datetime, timedelta) -> period w/ timedelta freq
        # (numeric, numeric) -> point sequence
        # (numeric, [str, timedelta]) -> INVALID
        if self.time_period is not None:
            start_time = self.time_period.start_time
            period_length = self.time_period.period_length
            error.value_check(
                "<COR36718278E>",
                start_time.time is not None,
                "start_time must be set in time_period",
            )
            error.value_check(
                "<COR36718279E>",
                period_length.time is not None,
                "period_length must be set in time_period",
            )

            numeric_start_time = start_time.ts_epoch is None
            numeric_period = period_length.dt_str is None and (
                period_length.dt_int is not None or period_length.dt_float is not None
            )
            error.value_check(
                "<COR36962854E>",
                not (numeric_start_time and not numeric_period),
                "Time period cannot have a numeric start_time with a timedelta period_length",
            )

            if numeric_start_time:
                df_kwargs["index"] = pd.RangeIndex(
                    start=start_time.time,
                    stop=period_length.time * num_rows,
                    step=period_length.time,
                )
            elif numeric_period:
                df_kwargs["index"] = pd.period_range(
                    start_time.ts_epoch.as_datetime(),
                    freq=timedelta(seconds=period_length.time),
                    periods=num_rows,
                )
            else:
                df_kwargs["index"] = pd.period_range(
                    start_time.ts_epoch.as_datetime(),
                    freq=period_length.dt_str,
                    periods=num_rows,
                )

        # Otherwise, interpret the sequence of time values directly
        else:
            time_points = self.time_points.points
            error.value_check(
                "<COR11757382E>",
                time_points is not None and len(time_points) == num_rows,
                "Number of time points {} doesn't match number of rows {}",
                -1 if time_points is None else len(time_points),
                num_rows,
            )
            if time_points:
                # Convert to a sequence of contiguous points
                time_point_values = [tp.time for tp in time_points]
                time_point_type = type(time_point_values[0])
                error.type_check_all(
                    "<COR79828262E>",
                    time_point_type,
                    time_point_values=time_point_values,
                )

                # If the type needs conversion to datetimes, do so
                if time_point_type == Seconds:
                    time_point_values = [val.as_datetime() for val in time_point_values]

                df_kwargs["index"] = time_point_values

        # Make the columns dict
        value_labels = self.value_labels or range(len(self.values))
        error.value_check(
            "<COR60320473E>",
            len(value_labels) == len(self.values),
            "Wrong number of value labels {} for {} value columns",
            len(value_labels),
            len(self.values),
        )

        def deserialize_values_if_necessary(sequence_values):
            if isinstance(sequence_values, ValueSequence.TimePointSequence):
                return [dateutil.parser.parse(v) for v in sequence_values.values]
            if isinstance(sequence_values, ValueSequence.AnyValueSequence):
                return [json.loads(v) for v in sequence_values.values]
            if isinstance(sequence_values, ValueSequence.VectorValueSequence):
                # this is required as the underlying type is just a repeated scalar field, we need
                # it to be a list
                return [list(v) for v in sequence_values.values]
            return sequence_values.values

        df_kwargs["data"] = dict(
            zip(
                value_labels,
                (deserialize_values_if_necessary(col.sequence) for col in self.values),
            )
        )

        result_df = pd.DataFrame(**df_kwargs)
        if self.timestamp_label != "":
            result_df.reset_index(inplace=True)
            result_df = result_df.rename(columns={"index": self.timestamp_label})

        # Not exposing the _single_timeseries and the dataframe will be cached elsewhere
        # self._backend = PandasTimeSeriesBackend(result_df, timestamp_column=self.timestamp_label,
        #   value_columns=value_labels)
        # Make the data frame
        return result_df, self.timestamp_label, value_labels

    def __len__(self) -> int:
        """Return the length of the single time series object.

        Returns:
            int: Length
        """
        if self.values:
            return len(self.values[0].sequence.values)
        return 0

    def __eq__(self, other: "SingleTimeSeries") -> bool:
        """Equivalence operator for SingleTimeSeries objects.

        Performs ordering of data based on timestamp_label prior to checking for equivalence. Relies
        on underlying pandas equivalence testing function `pd.testing.assert_frame_equal`.

        Args:
            other (SingleTimeSeries): SingleTimeSeries to test against.

        Returns:
            bool: True if the SingleTimeSeries are equivalent.
        """

        error.type_check("<COR98387946E>", SingleTimeSeries, other=other)

        if self.timestamp_label != other.timestamp_label:
            return False

        sort_columns = [self.timestamp_label] if self.timestamp_label else []

        try:
            pd.testing.assert_frame_equal(
                self.as_pandas().sort_values(by=sort_columns),
                other.as_pandas().sort_values(by=sort_columns),
            )
        except AssertionError:
            return False

        return True

    ## Views ##

    def _as_pandas_ops(self, adf, include_timestamps: Union[None, bool] = False):
        """operate on pandas-like object instead of strictly pandas"""
        backend_df = adf

        # if we want to include timestamps, but it is not already in the dataframe, we need to add
        if include_timestamps and self.timestamp_label is None:
            dftouse = backend_df.copy(deep=False)  # this does seem to be necessary
            dftouse[self.__class__._DEFAULT_TS_COL] = (
                list(range(len(dftouse)))
                if isinstance(dftouse, pyspark.pandas.DataFrame)
                else np.arange(len(dftouse))
            )
            return dftouse
        # if we do not want timestamps, but we already have them in the dataframe, we need to return
        # a view without timestamps
        if (
            include_timestamps is not None and not include_timestamps
        ) and self.timestamp_label is not None:
            return backend_df.loc[:, backend_df.columns != self.timestamp_label]

        return backend_df

    def as_pandas(self, include_timestamps: Optional[bool] = None) -> "pd.DataFrame":
        """Get the view of this timeseries as a pandas DataFrame

        Args:
            include_timestamps (bool, optional): Control the addition or removal of
            timestamps. True will include timestamps, generating if needed, while False will
            remove timestamps. Use None to returned what is available, leaving unchanged.
            Defaults to None.

        Returns:
            pd.DataFrame: The view of the data as a pandas DataFrame
        """
        backend_df = self._get_pd_df()[0]
        return self._as_pandas_ops(
            adf=backend_df, include_timestamps=include_timestamps
        )

    def as_spark(
        self, include_timestamps: Optional[bool] = None
    ) -> "pyspark.sql.DataFrame":
        """Get the view of this timeseries as a spark DataFrame

        Args:
            include_timestamps (bool, optional): Control the addition or removal of
            timestamps. True will include timestamps, generating if needed, while False will
            remove timestamps. Use None to returned what is available, leaving unchanged.
            Defaults to None.

        Returns:
            pyspark.sql.DataFrame: The view of the data as a spark DataFrame
        """
        if not HAVE_PYSPARK:
            raise NotImplementedError(
                "You must have pyspark installed for this to work!"
            )

        # Third Party
        # pylint: disable=import-outside-toplevel
        from pyspark.pandas import DataFrame as psdataframe
        from pyspark.sql import SparkSession

        # Local
        # pylint: disable=import-outside-toplevel
        from .backends._spark_backends import SparkTimeSeriesBackend

        # If there is a backend that knows how to do the conversion, use that
        backend = getattr(self, "_backend", None)
        if backend is not None and isinstance(backend, SparkTimeSeriesBackend):
            backend_df = backend._pyspark_df
            pandas_like: psdataframe = backend_df.pandas_api()
            timeseries_magic = self._as_pandas_ops(
                pandas_like, include_timestamps=include_timestamps
            )
            return timeseries_magic.to_spark()

        spark = SparkSession.builder.config(conf=sparkconf_local()).getOrCreate()
        return spark.createDataFrame(
            strip_periodic(
                input_df=self.as_pandas(include_timestamps=include_timestamps)
            )
        )
