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
Tests for the Timeseries data model object
"""

# Standard
from datetime import datetime, timezone
from typing import Iterable, Union
import datetime as dt
import json
import os
import traceback
import warnings

# Third Party
from pandas import RangeIndex
import dateutil
import numpy as np
import pandas as pd
import pyspark
import pytest

# Local
from caikit.core.data_model import ProducerId
from caikit.interfaces.ts.data_model import SingleTimeSeries
from caikit.interfaces.ts.data_model.backends._spark_backends import ensure_spark_cached
from caikit.interfaces.ts.data_model.backends.spark_util import iteritems_workaround
from caikit.interfaces.ts.data_model.backends.util import (
    pd_timestamp_to_seconds,
    strip_periodic,
)
from tests.interfaces.ts.data_model.util import create_extended_test_dfs, df_project
from tests.interfaces.ts.helpers import sslocal_fixture, test_log
import caikit.interfaces.ts.data_model as dm

warnings.filterwarnings("ignore", category=ResourceWarning)

test_log.setLevel("DEBUG")


keys = [["a", "a", "b"], ["c", "d", "e"]]

key_cols = {f"key_{i}": keys[i] for i in range(2)}

# Standard reusable value columns
value_cols = [range(3), np.arange(0, 1.5, 0.5)]
value_cols_dict = ({f"val_{i}": val_col for i, val_col in enumerate(value_cols)}, float)
value_rows = (list(zip(*value_cols)), float)
value_rows_str = ([(x[0], f"value:{x[1]}") for x in value_rows[0]], "string")
value_rows_any = ([(x[0], f"value:{x[1]}") for x in value_rows[0]], object)
value_rows_list = ([(x[0], [x[1], x[1], x[1]]) for x in value_rows[0]], object)
value_rows_period = (
    [
        (
            x[0],
            dt.datetime(year=2022, month=1, day=int(round(x[1] + 1))),
        )
        for x in value_rows[0]
    ],
    "datetime64[ns]",
)


# Timestamp range types
int_numeric_range = range(0, 30, 10)
float_numeric_range = np.arange(0, 1.5, 0.5)
date_range_regular = pd.date_range("2000", freq="D", periods=3)
date_range_irregular = pd.date_range("2000", freq="B", periods=3)
period_range_regular = pd.period_range("2000", freq="D", periods=3)
period_range_irregular = pd.period_range("2000", freq="B", periods=3)

# Reusable timestamp column name
# NOTE: This is _intentionally_ not the same as the default!
default_ts_col_name = "ts"

# All testable data frame configurations!
testable_data_frames = []
for ts_range in [
    int_numeric_range,
    float_numeric_range,
    date_range_regular,
    date_range_irregular,
    period_range_regular,
    period_range_irregular,
]:
    # Data column types
    for data_arg, data_type in [
        value_rows,
        value_cols_dict,
        value_rows_str,
        value_rows_any,
        value_rows_period,
        value_rows_list,
    ]:
        cur_df = pd.DataFrame(data_arg)

        for k, v in key_cols.items():
            cur_df[k] = v
        # be explicit about type otherwise goes to Any
        cur_df = cur_df.astype({cur_df.columns[1]: data_type})
        cur_df.columns = cur_df.columns.astype(str)

        testable_data_frames.append((cur_df, None, list(key_cols.keys())[0], None))
        for i in range(len(key_cols)):
            k = list(key_cols.keys())[: i + 1]
            # Add df w/ ts in index
            testable_data_frames.append((cur_df, None, k, None))

        # Add df w/ ts in column
        if isinstance(data_arg, dict):
            full_data_arg = dict(**{default_ts_col_name: ts_range}, **data_arg)
            ts_col_name = default_ts_col_name
        else:
            full_data_arg = [[ts] + list(vals) for ts, vals in zip(ts_range, data_arg)]
            ts_col_name = "0"
        cur_df = pd.DataFrame(full_data_arg)

        for k, v in key_cols.items():
            cur_df[k] = v
        cur_df.columns = cur_df.columns.astype(str)
        testable_data_frames.append(
            (cur_df, ts_col_name, list(key_cols.keys())[0], None)
        )

        # value column is specified
        if isinstance(data_arg, dict):
            value_keys = list(data_arg.keys())
            value_keys.append(None)
            for value_key in value_keys:
                for i in range(len(key_cols)):
                    k = list(key_cols.keys())[: i + 1]
                    # Add df w/ ts in index
                    testable_data_frames.append((cur_df, ts_col_name, k, value_key))
        # value column is unspecified
        else:
            for i in range(len(key_cols)):
                k = list(key_cols.keys())[: i + 1]
                # Add df w/ ts in index
                testable_data_frames.append((cur_df, ts_col_name, k, None))

        # just a simple test to include int keys
        cur_df["key_int_1"] = np.array([1, 1, 3], dtype=np.int32)
        cur_df["key_int_2"] = np.array([4, 5, 6], dtype=np.int32)
        testable_data_frames.append(
            (cur_df, ts_col_name, ["key_int_1", "key_int_2"], None)
        )

# replicate and extended the dataframes with pyspark.sql.DataFrame if needed
testable_data_frames = create_extended_test_dfs(testable_data_frames)


def get_df_len(df_in):
    if isinstance(df_in, pd.DataFrame):
        return len(df_in)
    else:
        return len(df_in.toPandas())


def get_col_list(df_in, col):
    if isinstance(df_in, pd.DataFrame):
        return df_in[col].values.tolist()
    else:
        return df_in.toPandas()[col].values.tolist()


@pytest.mark.filterwarnings(
    "ignore:'PYARROW_IGNORE_TIMEZONE' environment variable was not set.*",
    "ignore:`to_list` loads all data into the driver's memory.*:pyspark.pandas.utils.PandasAPIOnSparkAdviceWarning",
    "ignore:`to_numpy` loads all data into the driver's memory.*:pyspark.pandas.utils.PandasAPIOnSparkAdviceWarning",
    "ignore:In a future version of pandas, a length 1 tuple will be returned.*:",
)
@pytest.mark.parametrize("df_mts_data", testable_data_frames)
def test_timeseries_spark(df_mts_data):
    """Subset of test_timeseries_pd to exercise to_spark functionality of multi_timeseries"""
    df, ts_source, key_source, value_source = df_mts_data

    with ensure_spark_cached(df) as df:
        value_source = None if value_source is None else [value_source]

        mts = dm.TimeSeries(
            df,
            key_column=key_source,
            timestamp_column=ts_source,
            value_columns=value_source,
        )

        # make sure include_timestamps is working properly
        if ts_source is None:
            pdf = mts.as_spark().toPandas()
            assert (pdf.columns == df_project(df).columns).all()

            pdf = mts.as_spark(include_timestamps=False).toPandas()
            assert (pdf.columns == df_project(df).columns).all()

            pdf = mts.as_spark(include_timestamps=True).toPandas()
            for _, group in pdf.groupby(key_source):
                assert (
                    group["timestamp"].values
                    == np.arange(start=0, stop=get_df_len(group))
                ).all()
        else:
            pdf = mts.as_spark().toPandas()
            assert pdf.columns.tolist() == df_project(df).columns.tolist()
            pdf = mts.as_spark(include_timestamps=False).toPandas()
            assert pdf.columns.tolist() == [x for x in df.columns if x != ts_source]
            pdf = mts.as_spark(include_timestamps=True).toPandas()
            assert get_col_list(pdf, ts_source) == get_col_list(
                strip_periodic(df, create_copy=True), ts_source
            )


@pytest.mark.filterwarnings(
    "ignore:'PYARROW_IGNORE_TIMEZONE' environment variable was not set.*",
    "ignore:`to_list` loads all data into the driver's memory.*:pyspark.pandas.utils.PandasAPIOnSparkAdviceWarning",
    "ignore:`to_numpy` loads all data into the driver's memory.*:pyspark.pandas.utils.PandasAPIOnSparkAdviceWarning",
    "ignore:If `index_col` is not specified for `to_spark`, the existing index is lost when converting to Spark DataFrame.*:pyspark.pandas.utils.PandasAPIOnSparkAdviceWarning",
    "ignore:In a future version of pandas, a length 1 tuple will be returned.*:",
)
@pytest.mark.slow
@pytest.mark.parametrize("df_mts_data", testable_data_frames)
def test_timeseries_pd(df_mts_data):
    """Tests for TimeSeries objects backed by Pandas data frames. This test is
    parametrized over ALL different flavors of data frame since all layouts
    should behave the same!
    """
    df, ts_source, key_source, value_source = df_mts_data

    with ensure_spark_cached(df) as df:
        test_log.debug("Running test_timeseries_pd:\n%s", df)
        test_log.debug("ts_source: %s", ts_source)
        test_log.debug("key_source: %s", key_source)

        value_source = None if value_source is None else [value_source]

        mts = dm.TimeSeries(
            df,
            key_column=key_source,
            timestamp_column=ts_source,
            value_columns=value_source,
        )

        if isinstance(df, pd.DataFrame):
            assert mts.as_pandas() is df
        elif isinstance(df, pyspark.sql.DataFrame):
            #  best we can do is this
            assert mts.as_pandas().equals(df.toPandas())
        else:
            ...

        # make sure include_timestamps is working properly
        if ts_source is None:
            pdf = mts.as_pandas()
            assert (pdf.columns == df.columns).all()

            pdf = mts.as_pandas(include_timestamps=False)
            assert (pdf.columns == df.columns).all()

            pdf = mts.as_pandas(include_timestamps=True)
            for _, group in pdf.groupby(key_source):
                assert (
                    group["timestamp"].values
                    == np.arange(start=0, stop=get_df_len(group))
                ).all()
        else:
            pdf = mts.as_pandas()
            assert (pdf.columns == df.columns).all()

            pdf = mts.as_pandas(include_timestamps=False)
            assert pdf.columns.tolist() == [x for x in df.columns if x != ts_source]

            pdf = mts.as_pandas(include_timestamps=True)
            assert get_col_list(pdf, ts_source) == get_col_list(df, ts_source)

        # note: Added this check in to speed up CI build as we still get 100% coverage without it.
        #  having said that, this should be added to a cron-job as part of a nightly build
        if (
            isinstance(df, pd.DataFrame)
            or os.getenv("RUN_SPARKDF_SLOW_TESTS", "0") == "1"
        ):
            # Verify that json serialization round-trips
            json_repr = mts.to_json()
            json_round_trip = dm.TimeSeries.from_json(json_repr)
            try:
                assert check_df_mts_eq(df, json_round_trip, ts_source, key_source)

                # Verify that proto serialization round-trips
                proto_repr = mts.to_proto()
                proto_round_trip = dm.TimeSeries.from_proto(proto_repr)
                assert check_df_mts_eq(df, proto_round_trip, ts_source, key_source)

                # Verify that the original source can convert properly
                assert check_df_mts_eq(df, mts, ts_source, key_source)
            except:
                traceback.print_exc()
                assert False


def get_ts_sequence(
    df: pd.DataFrame,
    ts_source: Union[str, int],
    key_source: Union[str, Iterable[str], None],
    ids: Union[Iterable[str], None],
) -> pd.Series:
    df_new = df_project(df).copy()
    if key_source is not None:
        if isinstance(key_source, str):
            key_source = [key_source]
        for i in range(len(key_source)):
            df_new = df_new[df_new[key_source[i]] == ids[i]]
        """Helper to pull the sequence based on where the source is"""
    return (
        RangeIndex(start=0, stop=df_project(df).shape[0], step=1)
        if ts_source is None
        else iteritems_workaround(df_new[ts_source])
    )


def check_df_mts_eq(
    df: pd.DataFrame,
    mts: dm.TimeSeries,
    ts_source: Union[str, int],
    key_source: Union[str, Iterable[str]],
) -> bool:
    # test internal data
    for ts in mts.timeseries:
        res = check_df_ts_eq(df, ts, ts_source, key_source)
        if not res:
            return False

    return True


def check_df_ts_eq(
    df: pd.DataFrame,
    datamodel_ts: SingleTimeSeries,
    ts_source: Union[str, int],
    key_source,
) -> bool:
    """Helper to make sure the actual data in the data frame and the TimeSeries
    line up
    """

    ###################
    ## Time Sequence ##
    ###################

    ts_from_df = get_ts_sequence(df, ts_source, key_source, datamodel_ts.ids.values)
    if not datamodel_ts.time_sequence:
        test_log.debug("No valid time sequence!")
        return False
    if isinstance(ts_from_df.dtype, pd.PeriodDtype):
        # If it's a periodic index, the timeseries may hold this as either a
        # PeriodicTimeSequence (if the freq is regular) or a PointTimeSequence
        # (if the freq is irregular)
        if datamodel_ts.time_period:
            if not datamodel_ts.time_period.start_time.ts_epoch:
                test_log.debug("Start time for periodic not based in the epoch")
                return False
            if (
                datamodel_ts.time_period.start_time.ts_epoch.as_datetime().timestamp()
                != ts_from_df.iloc[0].start_time.timestamp()
            ):
                test_log.debug(
                    "Periodic time sequence start time mismatch: %s != %s",
                    datamodel_ts.time_period.start_time.ts_epoch.as_datetime(),
                    ts_from_df[0].start_time,
                )
                return False

            # The period may either be a string (pandas period notation) or a
            # number of seconds
            if datamodel_ts.time_period.period_length.dt_str:
                if (
                    datamodel_ts.time_period.period_length.dt_str
                    != ts_from_df.dtype.freq.name
                ):
                    test_log.debug(
                        "Period str duration mismatch: %s != %s",
                        datamodel_ts.time_period.period_length.dt_str,
                        ts_from_df.dtype.freq.name,
                    )
                    return False

            elif not datamodel_ts.time_period.period_length.dt_sec:
                test_log.debug("Period length for periodic not in seconds or str")
                return False
            elif (
                datamodel_ts.time_period.period_length.dt_sec.as_timedelta()
                != ts_from_df.dtype.freq.delta
            ):
                test_log.debug(
                    "Period length mismatch: %s != %s",
                    datamodel_ts.time_period.period_length.dt_sec.as_timedelta(),
                    ts_from_df.dtype.freq.delta,
                )
                return False
    elif isinstance(ts_from_df, RangeIndex):
        if datamodel_ts.time_period.start_time.ts_int is None:
            test_log.debug("Start time for periodic not based in the int")
            return False
        if datamodel_ts.time_period.start_time.ts_int != ts_from_df.start:
            test_log.debug(
                "Periodic time sequence start time mismatch: %s != %s",
                datamodel_ts.time_period.start_time.ts_int,
                ts_from_df.start,
            )
            return False

        # The period may either be a string (pandas period notation) or a
        # number of seconds
        if datamodel_ts.time_period.period_length.dt_int is not None:
            if datamodel_ts.time_period.period_length.dt_int != ts_from_df.step:
                test_log.debug(
                    "Period int duration mismatch: %s != %s",
                    datamodel_ts.time_period.period_length.dt_int,
                    ts_from_df.step,
                )
                return False
    # If not a periodic index, the dm representation is a sequence of points
    else:
        if not datamodel_ts.time_points:
            test_log.debug("Sequential sequence not represented as points")
            return False

        # Make sure the appropriate point types are used
        if len(datamodel_ts.time_points.points) != len(ts_from_df):
            test_log.debug(
                "Time point length mismatch: %d != %d",
                len(datamodel_ts.time_points.points),
                len(ts_from_df),
            )
            return False

        # Compare point values. We use view_point.time which will pull the
        # appropriate backing point type
        for i, (datamodel_point, df_point) in enumerate(
            zip(datamodel_ts.time_points.points, ts_from_df)
        ):
            test_log.debug(
                "Comparing TimePoints of type %s / %s",
                type(datamodel_point.time),
                type(df_point),
            )
            datamodel_time = datamodel_point.time
            if isinstance(datamodel_time, dm.Seconds):
                datamodel_time = datamodel_time.as_datetime()
            # direct comparison of datetime and np.datetime64 objects
            # are fraught with ambiguity. Consider this example:
            # dt = datetime(year=2000, month=1, day=1, second=0, microsecond=0)
            # npdt = np.datetime64(dt.isoformat())
            # npdt == dt # True
            # np.datetime64("2000-01-01T00:00:00.000000000") == dt # False !
            # np.datetime64("2000-01-01T00:00:00.000000000") == npdt # True !
            # Confusing to say the least
            datamodel_seconds = pd_timestamp_to_seconds(datamodel_time)
            df_seconds = pd_timestamp_to_seconds(df_point)
            if datamodel_seconds != df_seconds:
                test_log.debug(
                    "Point value mismatch: %s != %s delta is %s",
                    datamodel_seconds,
                    df_seconds,
                    datamodel_seconds - df_seconds,
                )
                return False

    ############
    ## Values ##
    ############

    df_val_cols = [
        val_label if val_label in df.columns else int(val_label)
        for val_label in datamodel_ts.value_labels or df.columns
    ]
    test_log.debug("df_val_cols: %s", df_val_cols)
    if len(df_val_cols) != len(datamodel_ts.values):
        test_log.debug("Value labels and value columns have mismatched length")
        return False

    for df_val_col_key, ts_val_seq in zip(df_val_cols, datamodel_ts.values):
        ts_vals = list(ts_val_seq.sequence.values)
        ids = datamodel_ts.ids.values
        df_new = df_project(df).copy()
        if isinstance(key_source, str):
            key_source = [key_source]
        for i in range(len(key_source)):
            df_new = df_new[df_new[key_source[i]] == ids[i]]
        df_val_col = df_new[df_val_col_key]
        if len(df_val_col) != len(ts_vals):
            test_log.debug("Column %s has length mismatch", df_val_col_key)
            return False

        # TODO: what about Any?
        #  We currently give back the serialized version when values is called, but should it be the deserialized???
        np_value_col = df_val_col.to_numpy()
        if ts_val_seq.val_any is not None:
            ts_vals = [json.loads(v) for v in ts_vals]
        if ts_val_seq.val_timepoint is not None:
            ts_vals = [np.datetime64(dateutil.parser.parse(v)) for v in ts_vals]

        # we have to test each separately since each is a vector
        if ts_val_seq.val_vector is not None:
            ts_vals = [v for v in ts_vals]
            if not len(np_value_col) == len(ts_vals):
                test_log.debug("vector lengths didn't match")
                return False
            for idx, ts_vals_val in enumerate(ts_vals):
                to_check = (
                    np_value_col[idx].tolist()
                    if isinstance(np_value_col[idx], np.ndarray)
                    else np_value_col[idx]
                )
                if not to_check == (
                    ts_vals_val.tolist()
                    if hasattr(ts_vals_val, "tolist")
                    else ts_vals_val
                ):
                    test_log.debug(
                        "Column %s has value mismatch: %s != %s",
                        df_val_col_key,
                        df_val_col,
                        ts_vals_val,
                    )
                    return False
        else:
            if not (np_value_col == ts_vals).all():
                test_log.debug(
                    "Column %s has value mismatch: %s != %s",
                    df_val_col_key,
                    df_val_col,
                    ts_vals,
                )
                return False

    return True


def _cmp(it1, it2):
    # oh the joys of having a language with no types
    try:
        np.testing.assert_equal(
            [pd_timestamp_to_seconds(x) for x in it1],
            [pd_timestamp_to_seconds(x) for x in it2],
        )
        return True
    except AttributeError:
        ...
    except ValueError:
        ...
    except AssertionError:
        return False

    if hasattr(it1, "to_numpy") and hasattr(it2, "to_numpy"):
        return (it1.to_numpy() == it2.to_numpy()).all()
    else:
        return (it1 == it2).all()


def compare_np(
    df: pd.DataFrame,
    np_view: np.ndarray,
    ts_source: Union[str, int],
    key_source: Union[str, Iterable[str]],
    value_source: Iterable[str],
    data_model_columns,  # columns in the data model. This is required as if we are coming from json or some other source, we will not have all of the columns pertaining to the original dataframe
) -> bool:
    """Compare the output numpy view to the input data frame. The following
    conventions should be true:

    1. The first column of the ndarray should be the time sequence
    2. The ndarray's dtype should be the "lowest common denominator" of the time
        sequence and value columns (e.g. object < float < int)
    """
    ts_range = get_ts_sequence(df, ts_source, None, None)

    # Make sure the time sequence on timestamp matches
    if isinstance(key_source, str):
        key_source = [key_source]

    if isinstance(value_source, str):
        value_source = [value_source]

    # ordering when dealing with partititioned spark dataframes is
    # not guaranteed so we'll need to do this:

    np_view_as_pandas = pd.DataFrame(
        columns=data_model_columns, data=np_view
    ).sort_values(by=key_source + [ts_source])
    if not _cmp(ts_range, np_view_as_pandas[ts_source]):
        test_log.debug(
            "Numpy ts sequence mismatch: %s != %s",
            ts_range,
            np_view_as_pandas[ts_source],
        )
        return False

    val_cols = [
        col
        for col in data_model_columns
        if col != ts_source and col in key_source or col in value_source
    ]
    np_val_rows = np_view_as_pandas[val_cols].to_numpy()

    if not _cmp(
        np_view_as_pandas[val_cols].to_numpy(), df_project(df)[val_cols].to_numpy()
    ):
        test_log.debug(
            "NP view data mismatch: %s != %s", np_val_rows, df_project(df)[value_source]
        )
        return False

    return True


def test_pd_timestamp_to_seconds():
    adate = datetime(2000, 1, 1, tzinfo=timezone.utc)
    assert pd_timestamp_to_seconds(pd.Period(adate.isoformat())) == adate.timestamp()
    assert (
        pd_timestamp_to_seconds(np.datetime64(datetime(2000, 1, 1, 0, 0, 0, 0)))
        == adate.timestamp()
    )
    assert pd_timestamp_to_seconds(adate) == adate.timestamp()
    assert pd_timestamp_to_seconds(adate.timestamp()) == adate.timestamp()
    assert pd_timestamp_to_seconds(pd.Timestamp(adate.isoformat())) == adate.timestamp()
    with pytest.raises(Exception):
        pd_timestamp_to_seconds([])


@pytest.fixture(scope="module")
def trivial_pandas_df():
    return pd.DataFrame(columns=["a", "b", "c"], data=[[1, 2, 3], [1, 4, 5]])


@pytest.fixture(scope="module")
def trivial_spark_df(trivial_pandas_df, sslocal_fixture):
    return sslocal_fixture.createDataFrame(trivial_pandas_df)


def test_multi_timeseries_raises_on_bad_input(trivial_pandas_df):
    # Local
    import caikit

    caikit.interfaces.ts.data_model.timeseries.HAVE_PYSPARK = False
    df = trivial_pandas_df
    ts = dm.TimeSeries(df, key_column="a")
    with pytest.raises(NotImplementedError):
        ts.as_spark()
    caikit.interfaces.ts.data_model.timeseries.HAVE_PYSPARK = True

    with pytest.raises(NotImplementedError):
        iteritems_workaround("foobar")


# this method could be called internally, we just want to guard for that
def test_multi_timeseries_bad_attribute(trivial_pandas_df):
    df = trivial_pandas_df
    ts = dm.TimeSeries(df, key_column="a")

    with pytest.raises(ValueError):
        ts._backend.get_attribute(ts, "bad_attribute")


def test_multi_timeseries_spark_bad_attribute(trivial_spark_df):
    df = trivial_spark_df
    ts = dm.TimeSeries(
        df,
        key_column="a",
    )

    with pytest.raises(ValueError):
        ts._backend.get_attribute(ts, "bad_attribute")


def test_as_spark_with_str_key_cols(trivial_spark_df):
    df = trivial_spark_df
    ts = dm.TimeSeries(
        df,
        key_column="a",
    )
    p1 = ts.as_spark(include_timestamps=True).toPandas()
    p2 = ts.as_pandas(include_timestamps=True)
    assert (p1.to_numpy() == p2.to_numpy()).all()


def test_as_spark_with_producer_id(trivial_spark_df):
    df = trivial_spark_df
    ts = dm.TimeSeries(
        df,
        key_column="a",
        producer_id=ProducerId("Test", "1.2.3"),
    )

    assert ts.producer_id.name == "Test"
    assert ts.producer_id.version == "1.2.3"


def test_mts_len(sslocal_fixture):
    df = pd.concat(
        [
            pd.DataFrame(
                [(x, "A", x * 5, x * 1.333) for x in range(10)],
                columns=["ts", "key", "val", "val2"],
            ),
            pd.DataFrame(
                [(x, "B", x * 5, x * 1.333) for x in range(30)],
                columns=["ts", "key", "val", "val2"],
            ),
        ]
    )

    # spark
    mts = dm.TimeSeries(
        sslocal_fixture.createDataFrame(df),
        timestamp_column="ts",
        key_column="key",
    )

    assert len(mts) == 40

    # pandas
    mts = dm.TimeSeries(
        df,
        timestamp_column="ts",
        key_column="key",
    )

    assert len(mts) == 40

    # no backend
    mts = dm.TimeSeries(
        df,
        timestamp_column="ts",
    )

    assert len(mts) == 40


@pytest.mark.filterwarnings(
    "ignore:.*loads all data into the driver's memory.*:pyspark.pandas.utils.PandasAPIOnSparkAdviceWarning",
    "ignore:toPandas attempted Arrow optimization.*:UserWarning",
)
def test_dm_serializes_spark_vectors(sslocal_fixture):
    # Standard
    from datetime import datetime

    # Third Party
    from pyspark.ml.linalg import Vectors, VectorUDT
    from pyspark.sql.types import (
        ArrayType,
        DateType,
        StringType,
        StructField,
        StructType,
    )

    v = Vectors.dense([1.0, 2.0])
    schema = StructType(
        [
            StructField("date", DateType(), True),
            StructField("id", StringType(), True),
            StructField("value", VectorUDT(), True),
        ]
    )

    data = [
        (datetime(year=2020, month=1, day=1), "id", v),
    ]
    df = sslocal_fixture.createDataFrame(data=data, schema=schema)

    mts = dm.TimeSeries(
        df,
        key_column="id",
        timestamp_column="date",
        value_columns=["value"],
    )

    # test round trip
    json_str = mts.to_json()
    mts2 = dm.TimeSeries.from_json(json_str)
    assert json_str == mts2.to_json()


def test_ts_eq():
    """Test time series equivalence"""
    df = pd.concat(
        [
            pd.DataFrame(
                [("a", x, x * 5) for x in range(20)], columns=["id", "ts", "val"]
            ),
            pd.DataFrame(
                [("b", x, x * 5) for x in range(30)], columns=["id", "ts", "val"]
            ),
        ],
        axis=0,
    )

    mts = dm.TimeSeries(df, key_column=["id"], timestamp_column="ts")
    mts_a = dm.TimeSeries(df[df.id == "a"], key_column=["id"], timestamp_column="ts")
    mts_missing_time = dm.TimeSeries(
        df[df.ts < 20], key_column=["id"], timestamp_column="ts"
    )

    # null is equal
    assert dm.TimeSeries(pd.DataFrame()) == dm.TimeSeries(pd.DataFrame())

    # trivially equal
    assert mts == mts
    assert mts_a == mts_a

    # number of ids different
    assert mts != mts_a

    # same number of ids, but different ones
    df_c = df.copy()
    df_c.loc[df_c["id"] == "b", "id"] = "c"
    mts_c = dm.TimeSeries(df_c, key_column=["id"], timestamp_column="ts")

    assert mts != mts_c

    # missing time points
    assert mts != mts_missing_time
