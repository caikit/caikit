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
"""Internal utilities for supporting backend implementations"""

# Standard
from datetime import datetime
from typing import Union

# Third Party
import numpy as np
import pandas as pd


def timezoneoffset(adatetime: datetime) -> int:
    """Returns the timezone offset (in seconds)
    for a given datetime object relative to the local
    system's time.

    Args:
        adatetime (datetime): a date of interest.

    Returns:
        int: offset in seconds (can be negative)
    """
    return (
        adatetime.timestamp()
        - datetime(
            year=adatetime.year,
            month=adatetime.month,
            day=adatetime.day,
            hour=adatetime.hour,
            minute=adatetime.minute,
            second=adatetime.second,
            microsecond=adatetime.microsecond,
        ).timestamp()
    )


def pd_timestamp_to_seconds(ts) -> float:
    """Extract the seconds-since-epoch representation of the timestamp

    NOTE: The pandas Timestamp.timestamp() function returns a different value
        than Timestamp.to_pydatetime().timestamp()! Since we want this to
        round-trip with python datetime, we want the latter. They both claim to
        be POSIX, so something is missing leap-something!
    """
    if isinstance(ts, pd.Period):
        return ts.to_timestamp().timestamp()  # no utc shift
    if isinstance(ts, np.datetime64):
        return ts.astype("datetime64[ns]").astype(float) / 1e9
    if isinstance(ts, datetime):
        return ts.timestamp()
    if isinstance(ts, (int, float, np.int32, np.int64, np.float32, np.float64)):
        return float(ts)
    raise ValueError(f"invalid type {type(ts)} for parameter ts.")


def strip_periodic(
    input_df: pd.DataFrame, ts_col_name: Union[str, None] = None, create_copy=True
) -> pd.DataFrame:
    """
    Removes **the first instance** of a periodic timestamp info
    (because spark doesn't like these when constructing a pyspark.sql.DataFrame.)
    If no periodic timestamp values can be found, input_df is returned as is.
    This method is always a no-op if input_df is not a native pandas.DataFrame.
    """

    if not isinstance(input_df, pd.DataFrame):
        return input_df

    # find location of period field
    try:
        index = (
            [type(x) for x in input_df.dtypes].index(pd.core.dtypes.dtypes.PeriodDtype)
            if ts_col_name is None
            else input_df.columns.to_list().index(ts_col_name)
        )
    except ValueError:
        index = -1

    df = input_df
    if index >= 0:
        df = input_df if not create_copy else input_df.copy(deep=False)
        # df.iloc[:, index]
        df[df.columns[index]] = [
            x.to_timestamp() if hasattr(x, "to_timestamp") else x
            for x in df.iloc[:, index]
        ]

    return df
