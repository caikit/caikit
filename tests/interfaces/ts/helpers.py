"""
Common test helpers
"""

# Standard
from functools import reduce
from typing import Union
import os
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
import caikit.interfaces.ts.data_model as dm

sslocal = SparkSession.builder.getOrCreate

## Global Config ###############################################################

test_log = alog.use_channel("TEST")

alog.configure(
    default_level=os.environ.get("LOG_LEVEL", "warning"),
    filters=os.environ.get("LOG_FILTERS", "py4j.java_gateway:error"),
    formatter="json" if os.environ.get("LOG_JSON", "").lower() == "true" else "pretty",
    thread_id=os.environ.get("LOG_THREAD_ID", "").lower() == "true",
)


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


@pytest.fixture
def sample_spark_df():
    """Pytest fixture for a self-enclosed spark data frame"""
    spark = sslocal()

    sample_spark_df = spark.createDataFrame(
        [
            Row(**{key: sample_data[key][idx] for key in sample_data})
            for idx in range(len(sample_data["key"]))
        ]
    )
    warnings.filterwarnings("ignore", category=ResourceWarning)
    yield sample_spark_df


@pytest.fixture
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
