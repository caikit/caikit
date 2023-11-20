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
Data model definitions for structures in the time series domain
"""

# ordering is important here to permit protobuf loading and dynamic
# `caikit.core` setup
# pylint: disable=wrong-import-order,wrong-import-position

# Local
# Import the protobufs
from .package import TS_PACKAGE
from .time_types import (
    PeriodicTimeSequence,
    PointTimeSequence,
    Seconds,
    TimeDuration,
    TimePoint,
    ValueSequence,
)

from ._single_timeseries import SingleTimeSeries  # isort:skip
from .timeseries import TimeSeries  # isort:skip
from .timeseries_evaluation import Id, EvaluationRecord, EvaluationResult  # isort:skip
