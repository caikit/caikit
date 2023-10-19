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
Tests for data model util
"""

# Third Party
import pandas as pd

# Local
from caikit.interfaces.ts.data_model import TimeSeries
from caikit.interfaces.ts.data_model.util import mts_equals


def test_mts_equals():
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

    mts = TimeSeries(df, key_column=["id"], timestamp_column="ts")
    mts_a = TimeSeries(df[df.id == "a"], key_column=["id"], timestamp_column="ts")
    mts_missing_time = TimeSeries(
        df[df.ts < 20], key_column=["id"], timestamp_column="ts"
    )

    # null is equal
    assert mts_equals(TimeSeries(pd.DataFrame()), TimeSeries(pd.DataFrame()))

    # trivially equal
    assert mts_equals(mts, mts)
    assert mts_equals(mts_a, mts_a)

    # number of ids different
    assert not mts_equals(mts, mts_a)

    # same number of ids, but different ones
    df_c = df.copy()
    df_c.loc[df_c["id"] == "b", "id"] = "c"
    mts_c = TimeSeries(df_c, key_column=["id"], timestamp_column="ts")

    assert not mts_equals(mts, mts_c)

    # missing time points
    assert not mts_equals(mts, mts_missing_time)
