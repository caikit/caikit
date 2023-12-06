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
from typing import Iterable, List, Optional, Tuple

# Third Party
import numpy as np
import pandas as pd

# First Party
import alog

# Local
from ....core import DataObjectBase
from ....core.data_model import ProducerId, dataobject
from ....core.exceptions import error_handler
from ._single_timeseries import SingleTimeSeries
from .backends.base import MultiTimeSeriesBackendBase
from .backends.pandas_backends import PandasMultiTimeSeriesBackend
from .backends.util import strip_periodic
from .package import TS_PACKAGE
from .toolkit.optional_dependencies import HAVE_PYSPARK, pyspark
from .toolkit.sparkconf import sparkconf_local

log = alog.use_channel("TSDM")
error = error_handler.get(log)


@dataobject(package=TS_PACKAGE)
class TimeSeries(DataObjectBase):
    timeseries: List[SingleTimeSeries]
    id_labels: List[str]
    producer_id: ProducerId

    _DEFAULT_ID_COL = "_TS_RESERVED"
    _DEFAULT_TS_COL = "timestamp"

    def __init__(self, *args, **kwargs):
        """Constructing a TimeSeries will currently delegate
        to either a pandas or spark dataframe backend depending
        on whether a native pandas or spark dataframe are passed for
        the first argument respectively.
        """

        if "timeseries" in kwargs:
            self.timeseries = None
            self.id_labels = None
            self.producer_id = None
            is_multi = True
            for k, v in kwargs.items():
                if k == "timeseries" and not isinstance(v, list):
                    is_multi = False
                    setattr(self, k, [v])
                else:
                    setattr(self, k, v)

            # if id_labels was never set, that means we have a single timeseries
            if not is_multi:
                self.id_labels = []
        else:
            error.value_check(
                "<COR81128386I>",
                len(args) != 0,
                "must have at least the data argument",
                args,
            )
            data_arg = args[0]

            # This will be done if SingleTimeSeries
            if kwargs.get("key_column") is None:
                kwargs["key_column"] = []

            if isinstance(data_arg, pd.DataFrame):
                self._backend = PandasMultiTimeSeriesBackend(*args, **kwargs)
            elif HAVE_PYSPARK and isinstance(data_arg, pyspark.sql.DataFrame):
                # Local
                # pylint: disable=import-outside-toplevel
                from ..data_model.backends._spark_backends import (
                    SparkMultiTimeSeriesBackend,
                )

                self._backend = SparkMultiTimeSeriesBackend(*args, **kwargs)

    def __len__(self) -> int:
        """Return the length of the time series object.

        Returns:
            int: Length
        """
        backend = getattr(self, "_backend", None)

        if backend is None:
            if self.timeseries:
                return sum(len(ts) for ts in self.timeseries)
            return 0

        if HAVE_PYSPARK:
            # Local
            # pylint: disable=import-outside-toplevel
            from ..data_model.backends._spark_backends import (
                SparkMultiTimeSeriesBackend,
            )

        if isinstance(backend, PandasMultiTimeSeriesBackend):
            return len(backend._df)
        if HAVE_PYSPARK and isinstance(self._backend, SparkMultiTimeSeriesBackend):
            return backend._pyspark_df.count()

        error.log_raise(
            "<COR75394521E>",
            f"Unknown backend {type(backend)}",
        )  # pragma: no cover

    def __eq__(self, other: "TimeSeries") -> bool:
        """Equivalence operator for TimeSeries objects.

        Args:
            other (TimeSeries): TimeSeries to test against.

        Returns:
            bool: True if the TimeSeries are equivalent.
        """

        # if number of mts is different, always unequal
        if len(self.timeseries) != len(other.timeseries):
            return False

        # empty mts is equal
        if len(self.timeseries) == 0:
            # ignoring edge cases of empty mts with different columns
            # unclear if this is even possible
            return True  # pragma: no cover

        # degenerate case
        if len(self.timeseries) == 1:
            return self.timeseries[0] == other.timeseries[0]

        # create map between keys and time series
        left_id_map = {tuple(ts.ids.values): ts for ts in self.timeseries}
        right_id_map = {tuple(ts.ids.values): ts for ts in other.timeseries}

        # quickly check keys are identical
        if set(left_id_map.keys()) != set(right_id_map.keys()):
            return False

        return all(l_ts == right_id_map[l_key] for l_key, l_ts in left_id_map.items())

    def _get_pd_df(self) -> Tuple[pd.DataFrame, Iterable[str], str, Iterable[str]]:
        """Convert the data to a pandas DataFrame, efficiently if possible"""

        # If there is a backend that knows how to do the conversion, use that
        backend = getattr(self, "_backend", None)
        if backend is not None and isinstance(backend, MultiTimeSeriesBackendBase):
            log.debug("Using backend pandas conversion")
            return backend.as_pandas()

        error.value_check(
            "<COR98388946E>",
            self.timeseries is not None,
            "Cannot create pandas data frame without any timeseries present",
        )

        error.value_check(
            "<COR59303952E>",
            self.id_labels is not None,
            "Cannot create pandas data frame without any key labels present",
        )

        key_columns = self.id_labels
        dfs = []
        value_columns = None
        timestamp_column = None
        for ts in self.timeseries:  # pylint: disable=not-an-iterable
            if value_columns is None:
                value_columns = ts.value_labels
                if ts.timestamp_label != "":
                    timestamp_column = ts.timestamp_label
            df = ts._get_pd_df()[0]

            for i, key_col in enumerate(key_columns):
                id_val = ts.ids.values[i]
                df[key_col] = [id_val] * df.shape[0]
            dfs.append(df)
        ignore_index = True  # timestamp_column != ""
        result = pd.concat(dfs, ignore_index=ignore_index)
        self._backend = PandasMultiTimeSeriesBackend(
            result,
            key_column=key_columns,
            timestamp_column=timestamp_column,
            value_columns=value_columns,
        )

        return (
            result,
            key_columns,
            timestamp_column,
            value_columns,
        )

    def as_pandas(
        self, include_timestamps: Optional[bool] = None, is_multi: Optional[bool] = None
    ) -> "pd.DataFrame":
        """Get the view of this timeseries as a pandas DataFrame

        Args:
            include_timestamps (bool, optional): Control the addition or removal of
            timestamps. True will include timestamps, generating if needed, while False will
            remove timestamps. Use None to returned what is available, leaving unchanged.
            Defaults to None.

            is_multi (bool, optional): Controls how id_labels are handled in the output. If
            the id_labels are specified in the data model, they are always returned. If there
            are no id_labels specified, setting is_multi to True will add a new column with
            generated id labels (0), while False or None will not add any id_labels.

        Returns:
            pd.DataFrame: The view of the data as a pandas DataFrame
        """
        # if as_pandas is_multi is True, and timeseries is_multi is False => add a RESERVED id
        #   column with constant value
        # if as_pandas is_multi is True, and timeseries is_multi is True => do nothing just return
        #   as is
        # if as_pandas is_multi is False, and timeseries is_multi is True => remove the id columns
        # if as_pandas is_multi is False, and timeseries is_multi is False => do nothing just
        #   return as is
        # if as_pandas is_multi is None => do nothing just return as is
        if len(self.id_labels) == 0:
            # pylint: disable=unsubscriptable-object
            df = self.timeseries[0].as_pandas(include_timestamps=include_timestamps)

            # add a RESERVED id column with constant value
            if is_multi is not None and is_multi:
                df = df.copy(deep=True)
                df[self.__class__._DEFAULT_ID_COL] = np.zeros(len(df), dtype=np.int32)
            return df

        backend_df = self._get_pd_df()[0]
        timestamp_column = self._backend._timestamp_column

        # if we want to include timestamps, but it is not already in the dataframe, we need to
        # add it
        if include_timestamps and timestamp_column is None:
            backend_df = backend_df.copy()  # avoid mutating original
            ts_column = self.__class__._DEFAULT_TS_COL
            backend_df[ts_column] = [0] * len(backend_df)
            backend_df[ts_column] = backend_df.groupby(
                self._backend._key_column, sort=False
            )[ts_column].transform(lambda x: list(range(len(x))))
            return backend_df
        # if we do not want timestamps, but we already have them in the dataframe, we need to
        # return a view without timestamps
        if (
            include_timestamps is not None and not include_timestamps
        ) and timestamp_column is not None:
            return backend_df.loc[:, backend_df.columns != timestamp_column]

        return backend_df

    def as_spark(
        self, include_timestamps: Optional[bool] = None, is_multi: Optional[bool] = None
    ) -> "pyspark.sql.DataFrame":
        """Get the view of this timeseries as a spark DataFrame

        Args:
            include_timestamps (bool, optional): Control the addition or removal of
            timestamps. True will include timestamps, generating if needed, while False will
            remove timestamps. Use None to returned what is available, leaving unchanged.
            Defaults to None.

            is_multi (bool, optional): Controls how id_labels are handled in the output. If
            the id_labels are specified in the data model, they are always returned. If there
            are no id_labels specified, setting is_multi to True will add a new column with
            generated id labels (0), while False or None will not add any id_labels.

        Returns:
            pyspark.sql.DataFrame: The view of the data as a spark DataFrame
        """
        if not HAVE_PYSPARK:
            raise NotImplementedError("pyspark must be available to use this method.")

        # todo: is this right???
        if len(self.id_labels) == 0:
            # pylint: disable=unsubscriptable-object
            df = self.timeseries[0].as_spark(include_timestamps=include_timestamps)
            # add a RESERVED id column with constant value
            if is_multi is not None and is_multi:
                df = df.pandas_api()
                df = df.copy(deep=True)
                df[self.__class__._DEFAULT_ID_COL] = np.zeros(
                    len(df), dtype=np.int32
                ).tolist()
                df = df.to_spark()
            return df

        # Third Party
        # pylint: disable=import-outside-toplevel
        from pyspark.sql import SparkSession

        # Local
        # pylint: disable=import-outside-toplevel
        from ..data_model.backends._spark_backends import SparkMultiTimeSeriesBackend

        # If there is a backend that knows how to do the conversion, use that
        backend = getattr(self, "_backend", None)
        if backend is not None and isinstance(backend, SparkMultiTimeSeriesBackend):
            answer = backend._pyspark_df
            timestamp_column = backend._timestamp_column
            if include_timestamps and timestamp_column is None:

                def append_timestamp_column(aspark_df, key_cols, timestamp_name):
                    sql = (
                        f"row_number() OVER (PARTITION BY {','.join(key_cols)} "
                        f"ORDER BY {','.join(key_cols)}) -1 as {timestamp_name}"
                    )
                    return aspark_df.selectExpr("*", sql)

                answer = append_timestamp_column(
                    answer, key_cols=self.id_labels, timestamp_name="timestamp"
                )
            elif (
                include_timestamps is not None
                and not include_timestamps
                and timestamp_column is not None
            ):
                answer = answer.drop(timestamp_column)
            return answer

        pdf = strip_periodic(
            self.as_pandas(include_timestamps=include_timestamps),
            create_copy=True,
        )
        return (
            SparkSession.builder.config(conf=sparkconf_local())
            .getOrCreate()
            .createDataFrame(pdf)
        )
