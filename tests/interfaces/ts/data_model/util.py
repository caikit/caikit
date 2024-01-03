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
"""Utilities used in data_model tests"""

# Standard
from typing import Iterable, List, Tuple
import copy
import os

# Third Party
import numpy as np
import pandas as pd
import pyspark

# Local
from caikit.interfaces.ts.data_model.backends.util import strip_periodic
from tests.interfaces.ts.helpers import sslocal


def _create_spark_dataframes(
    pandas_dfs: Iterable[Tuple],
) -> Iterable[Tuple]:
    """Creates spark dataframe versions of given native pandas dataframes."""

    spark_session: pyspark.sql.SparkSession = sslocal()
    answer = []
    for tup in pandas_dfs:
        dftouse = strip_periodic(tup[0], ts_col_name=tup[1])
        toappend = (
            (spark_session.createDataFrame(dftouse), tup[1], tup[2], tup[3])
            if len(tup) > 2
            else (spark_session.createDataFrame(dftouse), tup[1])
        )
        answer.append(toappend)
    return answer


def create_extended_test_dfs(testable_pandas_data_frames: List[Tuple]) -> List[Tuple]:
    """Extend (or not) the input list of native pandas dataframes with their spark datafram equivalents.
    Allow picking and choosing via an environment variable setting for DFTYPE."""
    answer = copy.copy(testable_pandas_data_frames)
    DFTYPE = os.getenv("DFTYPE", None)
    try:
        # only create the spark dataframes if needed
        if DFTYPE is None or DFTYPE == "spark_all":
            testable_spark_dataframes = _create_spark_dataframes(
                testable_pandas_data_frames
            )

        if DFTYPE is None:
            answer.extend(testable_spark_dataframes)
        elif DFTYPE == "pandas_all":
            ...  # no op
        elif DFTYPE == "spark_all":
            answer = testable_spark_dataframes
        # elif DFTYPE.startswith("pandas_"):
        #     answer = [answer[int(DFTYPE.split("_")[1])]]
        # elif DFTYPE.startswith("spark_"):
        #     answer = [testable_spark_dataframes[int(DFTYPE.split("_")[1])]]
        else:
            raise Exception(f"invalid setting {DFTYPE} for DFTYPE")
        return answer
    except IndexError as ie:
        print(ie)
        return testable_pandas_data_frames + testable_spark_dataframes


def key_helper(df, baskey):
    """This might not be necessary any more. Its intent was to enforce string
    column names for spark dataframes when we were allowing integer column names
    for the pandas backend implementation. It's kept for legacy purposes in the test
    for the time being."""
    return baskey if baskey is None or isinstance(df, pd.DataFrame) else str(baskey)


def df_project(df):
    """Return pandas api on the fly when needed by tests."""
    return df.pandas_api() if isinstance(df, pyspark.sql.DataFrame) else df


def df_col_to_nparray(df, col):
    if isinstance(df, pyspark.sql.DataFrame):
        return np.array([row[col] for row in df.collect()])

    return np.array(df[col].to_numpy())
