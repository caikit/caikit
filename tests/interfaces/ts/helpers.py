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
Common test helpers
"""

# Standard
from typing import Union
import warnings

# Third Party
from pyspark.sql import Row, SparkSession
import numpy as np
import pandas as pd
import pyspark
import pytest

# First Party
import alog

# Local
from caikit.interfaces.ts.data_model.toolkit import optional_dependencies
from caikit.interfaces.ts.data_model.toolkit.sparkconf import sparkconf_local
import caikit.interfaces.ts.data_model as dm

warnings.filterwarnings("ignore", category=ResourceWarning)

## Global Config ###############################################################

test_log = alog.use_channel("TEST")

## Test Data ###################################################################

sample_data = {"key": [1, 2, 3], "val": [4, 5, 6], "val2": [7.1, 8.1, 9.1]}
sample_df = pd.DataFrame(sample_data)
sample_np = np.array(
    [sample_data["key"], sample_data["val"], sample_data["val2"]]
).transpose()
sample_np_univariate = sample_np[:, [0, 1]]
sample_ts = list(zip(sample_data["key"], sample_data["val"]))
sample_mvts = list(
    zip(
        sample_data["key"],
        ((v1, v2) for v1, v2 in zip(sample_data["val"], sample_data["val2"])),
    )
)


## Helpers #####################################################################


@pytest.fixture(scope="session")
def sslocal_fixture():
    spark_session = SparkSession.builder.config(conf=sparkconf_local()).getOrCreate()
    yield spark_session
    spark_session.stop()


def sslocal():
    return SparkSession.builder.config(conf=sparkconf_local()).getOrCreate()


@pytest.fixture(scope="session")
def sample_spark_df(sslocal_fixture):
    """Pytest fixture for a self-enclosed spark data frame"""
    spark = sslocal_fixture

    sample_spark_df_ = spark.createDataFrame(
        [
            Row(**{key: val[idx] for key, val in sample_data.items()})
            for idx in range(sample_data["key"])
        ]
    )
    yield sample_spark_df_


@pytest.fixture(scope="session")
def sample_spark_df_univariate(sample_spark_df):
    """Pytest fixture for a self-enclosed spark data frame"""
    return sample_spark_df.select(["key", "val"])


@pytest.fixture
def no_pandas():
    """Fixture to simulate running without pandas installed"""
    current = optional_dependencies.HAVE_PANDAS
    HAVE_PANDAS = False
    yield
    HAVE_PANDAS = current


@pytest.fixture
def no_spark():
    """Fixture to simulate running without pyspark installed"""
    current = optional_dependencies.HAVE_PYSPARK
    optional_dependencies.HAVE_PYSPARK = False
    yield
    optional_dependencies.HAVE_PYSPARK = current


## other helpers


def get_anytimeseries_length(
    X: Union[pd.DataFrame, dm.TimeSeries, pyspark.sql.DataFrame]
) -> Union[None, int]:
    """Get the the length of any AnyTimeSeries object"""
    if isinstance(X, pd.DataFrame):
        return len(X)
    elif isinstance(X, dm.TimeSeries):
        return len(X)
    elif isinstance(X, pyspark.sql.DataFrame):
        return X.count()
    else:
        raise ValueError("Unknown time series type provided")
