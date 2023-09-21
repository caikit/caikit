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
