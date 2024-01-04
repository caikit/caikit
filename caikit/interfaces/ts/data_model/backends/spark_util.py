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
"""Internal utilities for supporting spark backend implementations"""

# Standard
from typing import Any, Iterable, List

# Third Party
import pandas as pd

# Local
from ..toolkit.optional_dependencies import HAVE_PYSPARK, pyspark


def iteritems_workaround(series: Any, force_list: bool = False) -> Iterable:
    """pyspark.pandas.Series objects do not support
    iteration. For native pandas.Series objects this
    function will be a no-op.

    For pyspark.pandas.Series or other iterable objects
    we try to_numpy() (unless force_list
    is true) and if that fails we resort to a to_list()

    """

    # check that we can convert
    if not hasattr(series, "to_list") and not hasattr(series, "to_numpy"):
        raise NotImplementedError(
            f"invalid typed {type(series)} passed for parameter series"
        )

    if isinstance(series, pd.Series):
        return series

    # handle an edge case of pyspark.ml.linalg.DenseVector
    if (
        HAVE_PYSPARK
        and isinstance(series, pyspark.pandas.series.Series)
        and isinstance(series[0], pyspark.ml.linalg.Vector)
    ):
        return [x.toArray().tolist() for x in series.to_numpy()]

    # note that we're forcing a list only if we're not
    # a native pandas series
    if force_list:
        return series.to_list()

    try:
        return series.to_numpy()
    except:  # noqa: E722
        return series.to_list()


def mock_pd_groupby(a_df_like, by: List[str], return_pandas_api=False):
    """Roughly mocks the behavior of pandas groupBy but on a spark dataframe."""

    distinct_keys = a_df_like.select(by).distinct().collect()
    for dkey in distinct_keys:
        adict = dkey.asDict()
        filter_statement = ""
        for k, v in adict.items():
            filter_statement += f" {k} == '{v}' and"
        if filter_statement.endswith("and"):
            filter_statement = filter_statement[0:-3]
        sub_df = a_df_like.filter(filter_statement)
        value = tuple(adict.values())
        value = value[0] if len(value) == 1 else value
        yield value, sub_df.pandas_api() if return_pandas_api else sub_df
