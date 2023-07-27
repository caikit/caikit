"""
Common data model containing all data structures that are passed in and out of
blocks.
"""

# ordering is important here to permit protobuf loading and dynamic
# `caikit.core` setup
# pylint: disable=wrong-import-order,wrong-import-position

# Standard
from typing import Union

# Third Party
import pandas as pd

# Local
# Import the protobufs
from . import protobufs
from .package import TS_PACKAGE

# Import core enums and add in from this data model
from caikit.core.data_model import enums

enums.import_enums(globals())


# Local
from .time_types import (
    PeriodicTimeSequence,
    PointTimeSequence,
    Seconds,
    TimeDuration,
    TimePoint,
    ValueSequence,
)

# Import producer and data streams from the core
from caikit.core.data_model import *

from ._single_timeseries import SingleTimeSeries  # isort:skip
from .timeseries import TimeSeries  # isort:skip
from .timeseries_evaluate import Id, EvaluationRecord, EvaluationResult  # isort:skip

# from ..tasks import (  # isort:skip
#     AnomalyDetectionTask,
#     EvaluationTask,
#     ForecastingTask,
#     TransformersTask,
# )
