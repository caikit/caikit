# Third Party
import pandas as pd

# Local
import caikit.interfaces.ts.data_model as dm


def mts_equals(left: dm.TimeSeries, right: dm.TimeSeries, **kwargs) -> bool:
    """Compare if two TimeSeries objects are equal
    Args:
        left (dm.TimeSeries): A multi-time series to compare.
        right (dm.TimeSeries): A multi-time series to compare.
    Returns:
        bool: True if they are identical
    """

    # if number of mts is different, always unequal
    if len(left.timeseries) != len(right.timeseries):
        return False

    # empty mts is equal
    if len(left.timeseries) == 0:
        # ignoring edge cases of empty mts with different columns
        # unclear if this is even possible
        return True  # pragma: no cover

    sort_columns = (
        [left.timeseries[0].timestamp_label]
        if left.timeseries[0].timestamp_label
        else []
    )

    # Degenerate Multi-TS, just use pandas
    if len(left.timeseries) == 1:
        try:
            pd.testing.assert_frame_equal(
                left.as_pandas().sort_values(by=sort_columns),
                right.as_pandas().sort_values(by=sort_columns),
                **kwargs,
            )
            return True
        except AssertionError:
            return False

    # Real Multi-TS, try not to use pandas on full MTS
    # must have ids

    # create map between keys and time series
    left_id_map = {tuple(ts.ids.values): ts for ts in left.timeseries}
    right_id_map = {tuple(ts.ids.values): ts for ts in right.timeseries}

    # quickly check keys are identical
    if set(left_id_map.keys()) != set(right_id_map.keys()):
        return False

    for key, val in left_id_map.items():
        l_ts = val
        r_ts = right_id_map[key]
        try:
            pd.testing.assert_frame_equal(
                l_ts.as_pandas().sort_values(by=sort_columns),
                r_ts.as_pandas().sort_values(by=sort_columns),
                **kwargs,
            )
        except AssertionError:
            return False

    return True
