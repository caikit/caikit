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
from datetime import timezone
from typing import Union
import datetime as dt
import json
import warnings

# Third Party
from pandas import RangeIndex
import dateutil
import numpy as np
import pandas as pd
import pyspark.sql
import pytest

# Local
from caikit.interfaces.ts.data_model._single_timeseries import SingleTimeSeries
from caikit.interfaces.ts.data_model.backends._spark_backends import (
    SparkTimeSeriesBackend,
    ensure_spark_cached,
)
from caikit.interfaces.ts.data_model.backends.spark_util import iteritems_workaround
from caikit.interfaces.ts.data_model.backends.util import (
    pd_timestamp_to_seconds,
    strip_periodic,
)
from tests.interfaces.ts.data_model.util import (
    create_extended_test_dfs,
    df_project,
    key_helper,
)
from tests.interfaces.ts.helpers import test_log
import caikit.interfaces.ts.data_model as dm

warnings.filterwarnings("ignore", category=ResourceWarning)

## Helpers #####################################################################

# There is a large variety in how the timestamps and columns can be represented
# in a pandas DataFrame, so we need to handle all of the cases:
#
# Timestamp Range Types:
#   - Int numeric
#   - Float numeric
#   - Date range on a regular timedelta (e.g. Day)
#   - Date range on an irregular frequence (e.g. Business Day)
#   - Periodic date on a regular timedelta (e.g. Day)
#   - Periodic date on an irregular frequence (e.g. Business Day)
#
# Timestamp Location:
#   - Column
#   - Index
#
# Value Columns:
#   - Numeric keys
#   - String keys

# Standard reusable value columns
value_cols = [range(3), np.arange(0, 1.5, 0.5)]
value_cols_dict = ({f"val_{i}": val_col for i, val_col in enumerate(value_cols)}, float)
value_rows = (list(zip(*value_cols)), float)
value_rows_str = ([(x[0], f"value:{x[1]}") for x in value_rows[0]], "string")
value_rows_any = ([(x[0], f"value:{x[1]}") for x in value_rows[0]], object)
value_rows_list = ([(x[0], [x[1], x[1], x[1]]) for x in value_rows[0]], object)
value_rows_period = (
    [
        (x[0], dt.datetime(year=2022, month=1, day=int(round(x[1] + 1))))
        for x in value_rows[0]
    ],
    "datetime64[ns]",
)


def get_ts_sequence(df: pd.DataFrame, ts_source: Union[str, int]) -> pd.Series:
    """Helper to pull the sequence based on where the source is"""
    key = key_helper(df, ts_source)
    return (
        RangeIndex(start=0, stop=df_project(df).shape[0], step=1)
        if key is None
        else df_project(df)[key]
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
        cur_df = pd.DataFrame(data_arg, index=ts_range)
        # be explicit about type otherwise goes to Any
        cur_df = cur_df.astype({cur_df.columns[1]: data_type})
        cur_df.columns = cur_df.columns.astype(str)

        # Add df w/ ts in index
        testable_data_frames.append(
            (
                cur_df,
                None,
            )
        )

        # Add df w/ ts in column
        if isinstance(data_arg, dict):
            full_data_arg = dict(**{default_ts_col_name: ts_range}, **data_arg)
            ts_col_name = default_ts_col_name
        else:
            full_data_arg = [[ts] + list(vals) for ts, vals in zip(ts_range, data_arg)]
            ts_col_name = "0"

        cur_df = pd.DataFrame(full_data_arg)
        cur_df.columns = cur_df.columns.astype(str)
        testable_data_frames.append((cur_df, ts_col_name))
        # let's append spark dataframes to this

original_length = len(testable_data_frames)
test_log.debug("Made a total of %d testable data frames!", original_length)

# replicate and extended the dataframes with pyspark.sql.DataFrame if needed
testable_data_frames = create_extended_test_dfs(testable_data_frames)


def check_df_ts_eq(
    df: pd.DataFrame,
    ts: SingleTimeSeries,
    ts_source: Union[str, int],
) -> bool:
    """Helper to make sure the actual data in the data frame and the TimeSeries
    line up
    """

    ###################
    ## Time Sequence ##
    ###################

    # some evaluations below require a pandas-like api
    dfeval = df_project(df)

    df_ts_range = get_ts_sequence(dfeval, ts_source)
    if not ts.time_sequence:
        test_log.debug("No valid time sequence!")
        return False
    if isinstance(df_ts_range.dtype, pd.PeriodDtype):
        # If it's a periodic index, the timeseries may hold this as either a
        # PeriodicTimeSequence (if the freq is regular) or a PointTimeSequence
        # (if the freq is irregular)
        if ts.time_period:
            if not ts.time_period.start_time.ts_epoch:
                test_log.debug("Start time for periodic not based in the epoch")
                return False
            if (
                ts.time_period.start_time.ts_epoch.as_datetime().timestamp()
                != df_ts_range[0].start_time.timestamp()
            ):
                test_log.debug(
                    "Periodic time sequence start time mismatch: %s != %s",
                    ts.time_period.start_time.ts_epoch.as_datetime(),
                    df_ts_range[0].start_time,
                )
                return False

            # The period may either be a string (pandas period notation) or a
            # number of seconds
            if ts.time_period.period_length.dt_str:
                if ts.time_period.period_length.dt_str != df_ts_range.dtype.freq.name:
                    test_log.debug(
                        "Period str duration mismatch: %s != %s",
                        ts.time_period.period_length.dt_str,
                        df_ts_range.dtype.freq.name,
                    )
                    return False

            elif not ts.time_period.period_length.dt_sec:
                test_log.debug("Period length for periodic not in seconds or str")
                return False
            elif (
                ts.time_period.period_length.dt_sec.as_timedelta()
                != df_ts_range.dtype.freq.delta
            ):
                test_log.debug(
                    "Period length mismatch: %s != %s",
                    ts.time_period.period_length.dt_sec.as_timedelta(),
                    df_ts_range.dtype.freq.delta,
                )
                return False
    elif isinstance(df_ts_range, RangeIndex):
        if ts.time_period.start_time.ts_int is None:
            test_log.debug("Start time for periodic not based in the int")
            return False
        if ts.time_period.start_time.ts_int != df_ts_range.start:
            test_log.debug(
                "Periodic time sequence start time mismatch: %s != %s",
                ts.time_period.start_time.ts_int,
                df_ts_range.start,
            )
            return False

        # The period may either be a string (pandas period notation) or a
        # number of seconds
        if ts.time_period.period_length.dt_int is not None:
            if ts.time_period.period_length.dt_int != df_ts_range.step:
                test_log.debug(
                    "Period int duration mismatch: %s != %s",
                    ts.time_period.period_length.dt_int,
                    df_ts_range.step,
                )
                return False
    # If not a periodic index, the dm representation is a sequence of points
    else:
        if not ts.time_points:
            test_log.debug("Sequential sequence not represented as points")
            return False

        # Make sure the appropriate point types are used
        if len(ts.time_points.points) != len(df_ts_range):
            test_log.debug(
                "Time point length mismatch: %d != %d",
                len(ts.time_points.points),
                len(df_ts_range),
            )
            return False

        # Compare point values. We use view_point.time which will pull the
        # appropriate backing point type
        for i, (datamodel_point, df_val) in enumerate(
            zip(ts.time_points.points, df_ts_range.to_list())
        ):
            test_log.debug(
                "Comparing TimePoints of type %s / %s",
                type(datamodel_point.time),
                type(df_val),
            )
            datamodel_val = datamodel_point.time
            if isinstance(datamodel_val, dm.Seconds):
                datamodel_val = datamodel_val.as_datetime()
            datamodel_seconds = pd_timestamp_to_seconds(datamodel_val)
            df_seconds = pd_timestamp_to_seconds(df_val)

            if datamodel_seconds != df_seconds:
                test_log.debug(
                    "Point value mismatch: %s != %s", datamodel_seconds, df_seconds
                )
                return False

    ############
    ## Values ##
    ############

    df_val_cols = [
        val_label if val_label in dfeval.columns else int(val_label)
        for val_label in ts.value_labels or dfeval.columns
    ]
    test_log.debug("df_val_cols: %s", df_val_cols)
    if len(df_val_cols) != len(ts.values):
        test_log.debug("Value labels and value columns have mismatched length")
        return False

    for df_val_col_key, ts_val_seq in zip(df_val_cols, ts.values):
        ts_vals = list(ts_val_seq.sequence.values)
        df_val_col = dfeval[df_val_col_key]
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
            for i in range(len(ts_vals)):
                # we can get ndarrays here as spark stores in ndarrays
                ts_to_check = (
                    ts_vals[i].tolist()
                    if isinstance(ts_vals[i], np.ndarray)
                    else ts_vals[i]
                )
                np_to_check = (
                    np_value_col[i].tolist()
                    if isinstance(np_value_col[i], np.ndarray)
                    else np_value_col[i]
                )
                if not ts_to_check == np_to_check:
                    test_log.debug(
                        "Column %s has value mismatch: %s != %s",
                        df_val_col_key,
                        df_val_col,
                        ts_vals,
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

    # ids is more thoroughly tested in MultiTimeSeries, where it is much more useful
    ids = ts.ids
    if ids is not None:
        return False

    return True


def compare_np(
    df: pd.DataFrame, np_view: np.ndarray, ts_source: Union[str, int]
) -> bool:
    """Compare the output numpy view to the input data frame. The following
    conventions should be true:

    1. The first column of the ndarray should be the time sequence
    2. The ndarray's dtype should be the "lowest common denominator" of the time
        sequence and value columns (e.g. object < float < int)
    """

    ts_sequence = get_ts_sequence(df, ts_source)

    # Make sure the time sequence matches
    # some equality operators are not supported down at the
    # java.util.ArrayList level (where pyspark.sql.DataFrames will go
    # down to)
    # so do it the old fashioned way
    for idx, value in enumerate(np_view[:, df.columns.get_loc(ts_source)]):
        if value != ts_sequence.iloc[idx]:
            test_log.debug(
                "Numpy ts sequence mismatch: %s != %s",
                ts_sequence[idx],
                value,
            )
            return False

    # Make sure the value sequences match
    df_np = pd.DataFrame(np_view)
    val_cols = [col for col in df.columns if col != ts_source]
    if ts_source is None:
        np_val_cols_len = len(df_np.columns)
    else:
        np_val_cols_len = len(df_np.columns) - 1

    np_val_rows = df_np[[df.columns.get_loc(c) for c in val_cols]].to_numpy()
    try:
        np.testing.assert_equal(
            np_val_rows.flatten(), df[val_cols].to_numpy().flatten()
        )
    except AssertionError as _:
        test_log.debug("NP view data mismatch: %s != %s", np_val_rows, df[val_cols])
        return False

    return True


## Tests #######################################################################


def test_not_serializable_value_val_any():
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    value_rows_not_serializable = [(x[0], Point(x[1], x[1])) for x in value_rows[0]]
    df = pd.DataFrame(value_rows_not_serializable, index=int_numeric_range)
    ts = dm.SingleTimeSeries(df)
    with pytest.raises(TypeError):
        ts.to_json()


@pytest.mark.filterwarnings(
    "ignore:'PYARROW_IGNORE_TIMEZONE' environment variable was not set.*",
    "ignore:`to_list` loads all data into the driver's memory.*:pyspark.pandas.utils.PandasAPIOnSparkAdviceWarning",
    "ignore:`to_numpy` loads all data into the driver's memory.*:pyspark.pandas.utils.PandasAPIOnSparkAdviceWarning",
    "ignore:If `index_col` is not specified for `to_spark`, the existing index is lost when converting to Spark DataFrame.*:pyspark.pandas.utils.PandasAPIOnSparkAdviceWarning",
)
def test_create_single_timeseries_dm():
    df = pd.DataFrame({"time_tick": [0, 1, 2], "value": [1.0, 2.0, 3.0]})
    ts = dm.SingleTimeSeries(df, timestamp_column="time_tick", value_columns=["value"])
    mts = dm.TimeSeries(timeseries=ts)

    assert mts.id_labels == []
    assert mts.timeseries[0].to_proto() == ts.to_proto()
    assert mts.timeseries[0].to_json() == ts.to_json()

    reserved_key = dm.TimeSeries._DEFAULT_ID_COL

    spark_mts = mts.as_spark(is_multi=True)
    assert isinstance(spark_mts, pyspark.sql.DataFrame)
    assert reserved_key in spark_mts.columns

    spark_ts = mts.as_spark()
    assert reserved_key not in spark_ts.columns

    pandas_mts = mts.as_pandas(is_multi=True)
    assert isinstance(pandas_mts, pd.DataFrame)
    assert reserved_key in pandas_mts.columns

    pandas_ts = mts.as_pandas()
    assert reserved_key not in pandas_ts.columns


def test_no_such_attribute_val_seq():
    value_rows_not_serializable = [(x[0], x[1]) for x in value_rows[0]]
    df = pd.DataFrame(value_rows_not_serializable, index=int_numeric_range)
    ts = dm.SingleTimeSeries(df)
    val_seq = ts.values[0]._backend
    with pytest.raises(AttributeError):
        val_seq.get_attribute(ts.values[0], "bad")


# todo
#  Looks like if we have dt_str and seconds, if we are not on a boundary, it gets truncated, that might be what we want
#  but can address later
start_times = [
    {"ts_epoch": {"seconds": 946702784.0}},
    {"ts_int": 946702784},
    {"ts_float": 946702784.0},
]
period_lengths = [
    {"dt_str": "D"},
    {"dt_int": 1},
    {"dt_float": 2.0},
    {"dt_sec": {"seconds": 3}},
]
periodic_time_seq_input = []
for start_time in start_times:
    for period_length in period_lengths:
        periodic_time_seq_input.append((start_time, period_length))


@pytest.mark.parametrize("input", periodic_time_seq_input)
def test_periodic_time_sequence_round_trip(input):
    start_time, period_length = input
    json_str = json.dumps({"startTime": start_time, "periodLength": period_length})
    periodic_time_sequence = dm.PeriodicTimeSequence.from_json(json_str)
    periodic_time_sequence_proto = periodic_time_sequence.to_proto()
    periodic_time_sequence = dm.PeriodicTimeSequence.from_proto(
        periodic_time_sequence_proto
    )

    # todo we need to handle this issue with camelcase being required for from_json
    assert (
        periodic_time_sequence.to_dict()["start_time"]
        == json.loads(json_str)["startTime"]
    )
    k = next(iter(period_length))
    assert (
        periodic_time_sequence.to_dict()["period_length"][k]
        == json.loads(json_str)["periodLength"][k]
    )


period_lengths = ["D", 1, 2.0, dm.Seconds.from_json(json.dumps({"seconds": 3}))]
results = ["D", 1, 2.0, {"seconds": 3}]


@pytest.mark.parametrize("input", period_lengths)
def test_time_duration_time_attribute(input):
    time_duration = dm.TimeDuration(time=input)
    assert time_duration.time == input


@pytest.mark.skip(
    "Raising an error for invalid oneof field values hasn't been implemented"
)
def test_time_duration_bad_attribute():
    with pytest.raises(AttributeError):
        _ = dm.TimeDuration(time=True)


time_points = [
    946702784,
    946702784.0,
    dm.Seconds.from_json(json.dumps({"seconds": 3})),
]


@pytest.mark.parametrize("input", time_points)
def test_time_point_time_attribute(input):
    time_point = dm.TimePoint(time=input)
    assert time_point.time == input


@pytest.mark.skip(
    "Raising an error for invalid oneof field values hasn't been implemented"
)
def test_time_point_time_bad_attribute():
    with pytest.raises(AttributeError):
        _ = dm.TimePoint(time=True)


@pytest.mark.skip(
    "Raising an error for invalid oneof field values hasn't been implemented"
)
def test_time_duration_never_set():
    with pytest.raises(AttributeError):
        _ = dm.TimeDuration()


def test_seconds():
    # setattr test
    seconds = dm.Seconds(seconds=1)
    assert dt.timedelta(seconds=1) == seconds.as_timedelta()

    # from timedelta
    seconds = dm.Seconds.from_timedelta(dt.timedelta(seconds=2))
    assert dt.timedelta(seconds=2) == seconds.as_timedelta()

    # from timestamp
    # Third Party
    import pytz

    seconds = dm.Seconds.from_datetime(dt.datetime(1990, 1, 1, tzinfo=timezone.utc))
    assert seconds.to_dict() == {"seconds": 631152000}


def test_empty_val_sequence():
    seq = dm.ValueSequence()
    assert seq.sequence is None


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
)
@pytest.mark.parametrize("df_ts_data", testable_data_frames)
def test_timeseries_pd(df_ts_data):
    """Tests for TimeSeries objects backed by Pandas data frames. This test is
    parametrized over ALL different flavors of data frame since all layouts
    should behave the same!
    """
    df, ts_source = df_ts_data

    with ensure_spark_cached(df) as df:
        # this doesn't for spark dataframes
        test_log.debug("Running test_timeseries_pd:\n%s", df)
        test_log.debug("ts_source: %s", ts_source)

        ts = dm.SingleTimeSeries(df, timestamp_column=ts_source)

        # Verify that the pandas view round-trips (and doesn't make a copy)
        # if we're using a spark backend, this is not a valid expectation
        # we need to check if ts_source isn't none as if it is none we will get a new dataframe that
        if not isinstance(ts._backend, SparkTimeSeriesBackend):
            assert ts.as_pandas() is df
        else:
            assert ts.as_pandas().equals(df.toPandas())

        # make sure include_timestamps is working properly
        if ts_source is None:
            pdf = ts.as_pandas()
            assert (pdf.columns == df.columns).all()

            pdf = ts.as_pandas(include_timestamps=False)
            assert (pdf.columns == df.columns).all()

            pdf = ts.as_pandas(include_timestamps=True)
            assert (
                pdf["timestamp"].values == np.arange(start=0, stop=get_df_len(df))
            ).all()
        else:
            pdf = ts.as_pandas()
            assert (pdf.columns == df.columns).all()

            pdf = ts.as_pandas(include_timestamps=False)
            assert pdf.columns.tolist() == [x for x in df.columns if x != ts_source]

            pdf = ts.as_pandas(include_timestamps=True)
            assert get_col_list(pdf, ts_source) == get_col_list(df, ts_source)

        # Verify that json serialization round-trips
        json_repr = ts.to_json()

        json_round_trip = dm.SingleTimeSeries.from_json(json_repr)
        assert check_df_ts_eq(df, json_round_trip, ts_source)

        json_obj = json.loads(json_repr)
        # Quick test to make sure that we can ingest json with start_time and period_length not being a pd.Series
        if json_obj.get("time_period"):
            json_obj["time_period"] = {
                "start_time": {"ts_int": 5},
                "period_length": {
                    "dt_int": 10,
                },
            }

            ts_new_period = dm.SingleTimeSeries.from_json(json_obj)
            assert ts_new_period.time_period.to_dict() == json_obj["time_period"]

            # static as it never changes here
            to_check = [5, 15, 25]

            # this is not a possible case, but checking for completeness
            if (
                ts_new_period.timestamp_label is None
                or ts_new_period.timestamp_label == ""
            ):
                assert ts_new_period.as_pandas().index.values.tolist() == to_check
            else:
                assert (
                    ts_new_period.as_pandas()[
                        ts_new_period.timestamp_label
                    ].values.tolist()
                    == to_check
                )

            # Verify that the pandas view looks the same if not from backend
            # assert check_df_ts_eq(ts_new_period.as_pandas(), ts_new_period, ts_source)

            json_obj["time_period"] = {
                "start_time": {"ts_epoch": {"seconds": 631195200}},
                "period_length": {
                    "dt_float": 3600.0,
                },
            }

            ts_new_period = dm.SingleTimeSeries.from_json(json_obj)
            assert ts_new_period.time_period.to_dict() == json_obj["time_period"]

            # static as it never changes here
            to_check = [
                pd.Period(value=dt.datetime.utcfromtimestamp(631195200), freq="H"),
                pd.Period(
                    value=dt.datetime.utcfromtimestamp(631195200 + 3600), freq="H"
                ),
                pd.Period(
                    value=dt.datetime.utcfromtimestamp(631195200 + 3600 * 2), freq="H"
                ),
            ]

            # this is not a possible case, but checking for completeness
            if (
                ts_new_period.timestamp_label is None
                or ts_new_period.timestamp_label == ""
            ):
                assert ts_new_period.as_pandas().index.values.tolist() == to_check
            else:
                assert (
                    ts_new_period.as_pandas()[
                        ts_new_period.timestamp_label
                    ].values.tolist()
                    == to_check
                )

            # Verify that the pandas view looks the same if not from backend
            # assert check_df_ts_eq(ts_new_period.as_pandas(), ts_new_period, ts_source)

        # Verify that proto serialization round-trips
        proto_repr = ts.to_proto()
        proto_round_trip = dm.SingleTimeSeries.from_proto(proto_repr)
        assert check_df_ts_eq(df, proto_round_trip, ts_source)

        # Verify that the pandas view looks the same if not from backend
        assert check_df_ts_eq(proto_round_trip.as_pandas(), ts, ts_source)


@pytest.mark.filterwarnings(
    "ignore:If `index_col` is not specified for `to_spark`, the existing index is lost when converting to Spark DataFrame.*:pyspark.pandas.utils.PandasAPIOnSparkAdviceWarning"
)
@pytest.mark.parametrize("df_ts_data", testable_data_frames)
def test_timeseries_spark(df_ts_data):
    """Tests for TimeSeries objects backed by Pandas data frames. This test is
    parametrized over ALL different flavors of data frame since all layouts
    should behave the same!
    """
    df, ts_source = df_ts_data

    with ensure_spark_cached(df) as df:
        # this doesn't for spark dataframes
        test_log.debug("Running test_timeseries_spark:\n%s", df)
        test_log.debug("ts_source: %s", ts_source)

        ts = dm.SingleTimeSeries(df, timestamp_column=ts_source)

        # Veryify that as_spark returns something the same as we passed in
        from_ts = ts.as_spark().toPandas().copy(deep=True)
        from_ts.reset_index(drop=True, inplace=True)
        from_df = (
            df.toPandas()
            if isinstance(ts._backend, SparkTimeSeriesBackend)
            else df.copy(deep=True)
        )
        from_df.reset_index(drop=True, inplace=True)
        from_df = strip_periodic(from_df)
        from_df_numpy = from_df.to_numpy()
        for idx, from_ts_val in enumerate(from_ts.to_numpy()):
            val_ts = (
                from_ts_val.tolist() if hasattr(from_ts_val, "tolist") else from_ts_val
            )
            val_df = (
                from_df_numpy[idx].tolist()
                if hasattr(from_df_numpy[idx], "tolist")
                else from_df_numpy[idx]
            )
            np.testing.assert_equal(val_ts, val_df)
            # assert val_ts[0] == val_df[0], idx
            # assert (val_ts[1] == val_df[1]).all(), idx

        # print(ts.as_spark().toPandas())
        # print(df_project(df))

        # make sure include_timestamps is working properly
        dftocompare = (
            df.toPandas()
            if isinstance(df, pyspark.sql.DataFrame)
            else strip_periodic(df, create_copy=True)
        )
        if ts_source is None:
            pdf = ts.as_spark().toPandas()
            assert (pdf.columns == dftocompare.columns).all()

            pdf = ts.as_spark(include_timestamps=False).toPandas()
            assert (pdf.columns == dftocompare.columns).all()

            pdf = ts.as_spark(include_timestamps=True).toPandas()
            assert (
                pdf["timestamp"].values
                == np.arange(start=0, stop=get_df_len(dftocompare))
            ).all()
        else:
            pdf = ts.as_spark().toPandas()
            assert (pdf.columns == dftocompare.columns).all()

            pdf = ts.as_spark(include_timestamps=False).toPandas()
            assert pdf.columns.tolist() == [
                x for x in dftocompare.columns if x != ts_source
            ]

            pdf = ts.as_spark(include_timestamps=True).toPandas()
            assert get_col_list(pdf, ts_source) == get_col_list(dftocompare, ts_source)


def test_timeseries_raises_on_bad_input():
    # Local
    import caikit

    with pytest.raises(NotImplementedError):
        ts = dm.SingleTimeSeries([])

    class Dummy:
        def to_list(self):
            return []

    assert [] == iteritems_workaround(Dummy(), force_list=False)

    caikit.interfaces.ts.data_model._single_timeseries.HAVE_PYSPARK = False

    df = pd.DataFrame([1, 2, 3])
    ts = dm.SingleTimeSeries(df)
    with pytest.raises(NotImplementedError):
        ts.as_spark()

    caikit.interfaces.ts.data_model._single_timeseries.HAVE_PYSPARK = True
