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
Core data model backends backed by pandas
"""

# Standard
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable, Optional, Tuple, Type, Union
import json

# Third Party
from pandas import RangeIndex
import numpy as np
import pandas as pd

# First Party
import alog

# Local
from .....core.data_model import DataBase, ProducerId
from .....core.exceptions import error_handler
from ... import data_model as dm
from .. import time_types
from ..toolkit.optional_dependencies import HAVE_PYSPARK
from .base import (
    MultiTimeSeriesBackendBase,
    StrictFieldBackendMixin,
    TimeSeriesBackendBase,
    UncachedBackendMixin,
)
from .spark_util import iteritems_workaround
from .util import pd_timestamp_to_seconds

if TYPE_CHECKING:
    # Local
    from ..timeseries import TimeSeries

log = alog.use_channel("PDBCK")
error = error_handler.get(log)


class PandasMultiTimeSeriesBackend(MultiTimeSeriesBackendBase):
    def as_pandas(self) -> Tuple[pd.DataFrame, Iterable[str], str, Iterable[str]]:
        return self._df, self._key_column, self._timestamp_column, self._value_columns

    def __init__(
        self,
        data_frame: pd.DataFrame,
        key_column: Union[Iterable[str], str],
        timestamp_column: Optional[str] = None,
        value_columns: Optional[Iterable[str]] = None,
        ids: Optional[Union[Iterable[int], Iterable[str]]] = None,
        producer_id: Optional[Union[Tuple[str, str], ProducerId]] = None,
    ):
        error.type_check("<COR81128390E>", pd.DataFrame, data_frame=data_frame)
        error.type_check(
            "<COR81128391E>",
            list,
            str,
            key_column=key_column,
        )
        error.type_check(
            "<COR81128392E>", str, int, type(None), timestamp_column=timestamp_column
        )
        error.type_check_all(
            "<COR81128393E>",
            str,
            int,
            allow_none=True,
            value_columns=value_columns,
        )
        error.type_check_all(
            "<COR81128394E>",
            str,
            allow_none=True,
            ids=ids,
        )
        error.type_check(
            "<COR81128395E>",
            tuple,
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )

        # Validate the column names
        error.value_check(
            "<COR04942296E>",
            (timestamp_column is None or (timestamp_column in data_frame.columns)),
            "Invalid timestamp column/index: {}",
            timestamp_column,
        )

        self._df = data_frame
        self._key_column = key_column
        self._timestamp_column = timestamp_column
        key_column_list = [key_column] if isinstance(key_column, str) else key_column
        # pylint: disable=duplicate-code
        self._value_columns = value_columns or [
            col
            for col in data_frame.columns
            if col != timestamp_column and col not in key_column_list
        ]
        self._ids = [] if ids is None else ids
        self._producer_id = (
            producer_id
            if isinstance(producer_id, ProducerId)
            else (ProducerId(*producer_id) if producer_id is not None else None)
        )
        self._timeseries = None
        self._key_columns = (
            [self._key_column]
            if isinstance(self._key_column, str)
            else self._key_column
        )

    def get_attribute(self, data_model_class: Type["TimeSeries"], name: str) -> Any:
        if name == "timeseries":
            result = []

            if len(self._key_columns) == 0:
                backend = PandasTimeSeriesBackend(
                    self._df,
                    timestamp_column=self._timestamp_column,
                    value_columns=self._value_columns,
                )
                result.append(dm.SingleTimeSeries(_backend=backend))
            else:
                for k, k_df in self._df.groupby(
                    self._key_columns
                    if len(self._key_columns) > 1
                    else self._key_columns[0]
                ):
                    # if it is a single key string, we want to just wrap it in a list
                    if isinstance(k, (str, int)):
                        k = [k]
                    backend = PandasTimeSeriesBackend(
                        k_df,
                        timestamp_column=self._timestamp_column,
                        value_columns=self._value_columns,
                        ids=k,
                    )
                    result.append(dm.SingleTimeSeries(_backend=backend))

            return result

        if name == "id_labels":
            return self._key_columns

        # If requesting producer_id or ids, just return the stored value
        if name == "producer_id":
            return self._producer_id

        raise ValueError(f"Provided an attribute name that does not exist - {name}")


class PandasTimeSeriesBackend(TimeSeriesBackendBase):
    """The PandasTimeSeriesBackend is responsible for managing the standard
    in-memory representation of a TimeSeries
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        timestamp_column: str = None,
        value_columns: Optional[Iterable[str]] = None,
        ids: Optional[Union[Iterable[int], Iterable[str]]] = None,
    ):
        """At init time, hold onto the data frame as well as the arguments that
        tell where the time and values live

        Args:
            data_frame:  pd.DataFrame
                The raw data frame
            timestamp_column:  Optional[str]
                The name of the column holding the timestamps. If set to None, timestamps will be
                assigned based on the rows index (default is None)
            value_columns:  Optional[Iterable[str]]
                A sequence of names of columns to hold as values
            ids:  Optional[iterable[int]]
                A sequence of numeric IDs associated with this TimeSeries
        """
        # Validate the types and column names
        error.type_check("<COR81128380E>", pd.DataFrame, data_frame=data_frame)
        error.type_check(
            "<COR81128381E>", str, type(None), timestamp_column=timestamp_column
        )
        error.type_check_all(
            "<COR81128382E>",
            str,
            allow_none=True,
            value_columns=value_columns,
        )
        error.type_check_all(
            "<COR81128383E>",
            str,
            np.int_,
            int,
            allow_none=True,
            ids=ids,
        )

        # Validate the column names

        error.value_check(
            "<COR81128385E>",
            (timestamp_column is None or (timestamp_column in data_frame.columns)),
            "Invalid timestamp column/index: {}",
            timestamp_column,
        )
        value_columns = value_columns or [
            col for col in data_frame.columns if col != timestamp_column
        ]
        error.value_check(
            "<COR89526927E>",
            # TODO: Support lambdas!
            all(value_col in data_frame.columns for value_col in value_columns),
            "Invalid value columns: {}",
            value_columns,
        )

        self._df = data_frame
        self._timestamp_column = timestamp_column
        self._value_columns = value_columns
        self._ids = [] if ids is None else ids

    # pylint: disable=too-many-return-statements
    def get_attribute(
        self,
        data_model_class: Type["dm.SingleTimeSeries"],
        name: str,
        external_df: pd.DataFrame = None,
    ) -> Any:
        """When fetching a data attribute from the timeseries, this aliases to
        the appropriate set of backend wrappers for the various fields.
        """

        # use the external definition of our pandas-like dataframe if
        # requested
        pandas_impl = external_df if external_df is not None else self._df

        if name == "timestamp_label":
            return self._timestamp_column

        if name == "ids" and self._ids is not None and len(self._ids) != 0:
            if isinstance(self._ids[0], (np.int_, int)):
                val = data_model_class.IntIDSequence(
                    values=[
                        id.item() if isinstance(id, np.int_) else id for id in self._ids
                    ]
                )
                return DataBase.OneofFieldVal(val=val, which_oneof="id_int")
            if isinstance(self._ids[0], str):
                val = data_model_class.StringIDSequence(values=self._ids)
                return DataBase.OneofFieldVal(val=val, which_oneof="id_str")

        # If requesting the value_labels, this is the value column names
        if name == "value_labels":
            return [str(val) for val in self._value_columns]

        # If requesting the "time_sequence" or one of the oneof fields, extract
        # the timestamps from the dataframe
        if name == "time_sequence":
            if self._timestamp_column is None:
                time_sequence = RangeIndex(start=0, stop=pandas_impl.shape[0], step=1)
            else:
                time_sequence = pandas_impl[self._timestamp_column]

            # If the sequence is periodic, use the PeriodicTimeSequence backend
            is_periodic = isinstance(time_sequence.dtype, pd.PeriodDtype) or isinstance(
                time_sequence, RangeIndex
            )
            if is_periodic:
                val = time_types.PeriodicTimeSequence.from_backend(
                    PandasPeriodicTimeSequenceBackend(time_sequence)
                )
                return DataBase.OneofFieldVal(val=val, which_oneof="time_period")
            # Otherwise, use the PointTimeSequence backend
            val = time_types.PointTimeSequence.from_backend(
                PandasPointTimeSequenceBackend(time_sequence)
            )
            return DataBase.OneofFieldVal(val=val, which_oneof="time_points")

        # If requesting the value sequences, return the wrapped value columns
        if name == "values":
            return [
                time_types.ValueSequence.from_backend(
                    PandasValueSequenceBackend(pandas_impl, col_name)
                )
                for col_name in self._value_columns
            ]

    def as_pandas(self) -> Tuple[pd.DataFrame, str, Iterable[str]]:
        """Return the underlying data frame"""
        return self._df, self._timestamp_column, self._value_columns


class PandasValueSequenceBackend(UncachedBackendMixin, StrictFieldBackendMixin):
    """Backend for ValueSequence backed by a set of columns in a Pandas
    DataFrame
    """

    @staticmethod
    def _serialize_any(any_val):
        try:
            json_str = json.dumps(any_val)
            return json_str
        except Exception as exc:
            raise TypeError("could not serialize the given value") from exc

    # This dtype is what shows up for non-periodic date ranges
    _TIMESTAMP_DTYPE = np.dtype("datetime64[ns]")

    # What types do we consider to be vector types
    _DEFAULT_VECTOR_TYPES = [list, np.ndarray]
    if HAVE_PYSPARK:
        # pyspark.pandas.DataFrame objects can contain
        # pyspark specific objects

        # Third Party
        # pylint: disable=import-outside-toplevel
        from pyspark.ml.linalg import Vector as SVector

        _DEFAULT_VECTOR_TYPES.append(SVector)

    def __init__(self, data_frame: pd.Series, col_name: str):
        """Initialize with the data frame and the value column name"""
        self._df = data_frame
        self._col_name = col_name
        # Determine which of the oneof types is valid for this sequence
        self._dtype = self._df[self._col_name].dtype
        self._converter = lambda x: x
        if str(self._dtype).startswith(
            str(self.__class__._TIMESTAMP_DTYPE)[:-1]
        ) or isinstance(self._dtype, pd.PeriodDtype):
            # what do we want to do here, are we just assuming it will convert forever?
            self._sequence_type = time_types.ValueSequence.TimePointSequence
            self._valid_oneof = "val_timepoint"
        # todo not sure why np.issubdtype is running into issue if this is run after
        elif self._dtype == "string":
            self._sequence_type = time_types.ValueSequence.StrValueSequence
            self._valid_oneof = "val_str"
        elif np.issubdtype(self._dtype, np.integer):
            self._sequence_type = time_types.ValueSequence.IntValueSequence
            self._valid_oneof = "val_int"
        elif np.issubdtype(self._dtype, np.floating):
            self._sequence_type = time_types.ValueSequence.FloatValueSequence
            self._valid_oneof = "val_float"
        # todo do we handle ndarrays in cells, if so we need to convert to list before going to json
        # as ndarray is not serializable
        # this is making the assumption that we have at least one value in the dataframe
        elif str(self._dtype) == "object" and isinstance(
            self._df[self._col_name].iloc[0],
            tuple(PandasValueSequenceBackend._DEFAULT_VECTOR_TYPES),
        ):
            self._sequence_type = time_types.ValueSequence.VectorValueSequence
            self._valid_oneof = "val_vector"
        else:
            self._sequence_type = time_types.ValueSequence.AnyValueSequence
            self._valid_oneof = "val_any"

    def get_attribute(
        self,
        data_model_class: Type[time_types.ValueSequence],
        name: str,
    ) -> Any:
        """Get the known attributes from the underlying DataFrame columns"""

        if name == "sequence":
            name = self._valid_oneof
        if name == self._valid_oneof and name in [
            "val_int",
            "val_float",
            "val_str",
            "val_vector",
        ]:
            return self._sequence_type(
                values=[
                    self._converter(val)
                    for val in iteritems_workaround(
                        self._df[self._col_name], force_list=True
                    )
                ],
            )
        if name == self._valid_oneof == "val_any":
            return self._sequence_type(
                values=[
                    self._serialize_any(val)
                    for val in iteritems_workaround(
                        self._df[self._col_name], force_list=False
                    )
                ]
            )

        if name == self._valid_oneof == "val_timepoint":
            return self._sequence_type(
                values=[
                    val.isoformat() if hasattr(val, "isoformat") else str(val)
                    for val in iteritems_workaround(
                        self._df[self._col_name], force_list=False
                    )
                ]
            )

        # Delegate to common parent logic
        return super().get_attribute(data_model_class, name)


class PandasPeriodicTimeSequenceBackend(UncachedBackendMixin, StrictFieldBackendMixin):
    """Backend for PeriodicTimeSequence backed by a Pandas Time Span"""

    def __init__(self, time_sequence):
        """Initialize with a periodic time sequence"""
        self._is_range_index = isinstance(time_sequence, RangeIndex)
        if self._is_range_index:
            self._start_time = time_sequence.start
            self._period_length = time_sequence.step
        else:
            self._start_time = (
                None if time_sequence.empty else time_sequence.iloc[0].start_time
            )
            self._period_length = time_sequence.dtype.freq.name

    def get_attribute(
        self,
        data_model_class: Type[time_types.PeriodicTimeSequence],
        name: str,
    ) -> Any:
        """Get the known attributes from the backend data"""
        if name == "start_time" and self._start_time is not None:
            return time_types.TimePoint.from_backend(
                PandasTimePointBackend(self._start_time)
            )
        if name == "period_length":
            if self._is_range_index:
                return time_types.TimeDuration(dt_int=self._period_length)

            return time_types.TimeDuration(dt_str=self._period_length)

        # Delegate to common parent logic
        return super().get_attribute(data_model_class, name)


class PandasPointTimeSequenceBackend(
    UncachedBackendMixin,
    StrictFieldBackendMixin,
):  # TODO: Should we cache this one???
    """Backend for PointTimeSequence backed by a Pandas Series"""

    def __init__(self, time_sequence: pd.Series):
        """Initialize with a series based time sequence"""
        self._time_sequence = time_sequence

    def get_attribute(
        self,
        data_model_class: Type[time_types.PointTimeSequence],
        name: str,
    ) -> Any:
        """Get the known attributes from the backend data"""
        if name == "points":
            # TODO: a user may have ints/floats stored as objects in their dataframe, should we
            # handle that or throw an exception
            return [
                time_types.TimePoint.from_backend(PandasTimePointBackend(point_data))
                for point_data in iteritems_workaround(
                    self._time_sequence, force_list=True
                )
            ]

        # Delegate to common parent logic
        return super().get_attribute(data_model_class, name)


class PandasTimePointBackend(UncachedBackendMixin, StrictFieldBackendMixin):
    """Backend for time point data held by Pandas"""

    def __init__(self, point_data: Any):
        """Initialize with the raw pandas value"""
        self._point_data = point_data

    def get_attribute(
        self,
        data_model_class: Type[time_types.TimePoint],
        name: str,
    ) -> Any:
        """Get the appropriate fields based on the data type of the point"""
        int_ok = name in ["time", "ts_int"]
        float_ok = name in ["time", "ts_float"]
        epoch_ok = name in ["time", "ts_epoch"]

        if epoch_ok and isinstance(
            self._point_data, (pd.Timestamp, datetime, np.datetime64, pd.Period)
        ):
            return time_types.Seconds(seconds=pd_timestamp_to_seconds(self._point_data))
        dtype = getattr(self._point_data, "dtype", None)
        if int_ok and (
            isinstance(self._point_data, int) or np.issubdtype(dtype, np.integer)
        ):
            return self._point_data
        if float_ok and (
            isinstance(self._point_data, float)
            or (dtype is not None and np.issubdtype(dtype, np.floating))
        ):
            return self._point_data
