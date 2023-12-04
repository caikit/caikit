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

# Third Party
import pandas as pd

# First Party
import alog

# Local
from ....core.exceptions import error_handler
from . import TimeSeries

log = alog.use_channel("TSDM")
error = error_handler.get(log)


# pylint: disable=too-many-return-statements
def mts_equals(left: TimeSeries, right: TimeSeries, **kwargs) -> bool:
    """Compare if two TimeSeries objects are equal
    Args:
        left (dm.TimeSeries): A multi-time series to compare.
        right (dm.TimeSeries): A multi-time series to compare.
    Returns:
        bool: True if they are identical
    """

    error.type_check("<COR98387946E>", TimeSeries, left=left, right=right)

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
