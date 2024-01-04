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
All timeseries tasks used by modules
"""

# Local
from .data_model.timeseries import TimeSeries
from .data_model.timeseries_evaluation import EvaluationResult
from caikit.core import TaskBase, task


@task(
    required_parameters={"X": TimeSeries},
    output_type=TimeSeries,
)
class AnomalyDetectionTask(TaskBase):
    """Task for all anomaly detection modules"""


@task(
    required_parameters={"targets": TimeSeries, "predictions": TimeSeries},
    output_type=EvaluationResult,
)
class EvaluationTask(TaskBase):
    """Task for all evaluation modules"""


@task(
    required_parameters={"X": TimeSeries},
    output_type=TimeSeries,
)
class ForecastingTask(TaskBase):
    """Task for all forecasting modules"""


@task(
    required_parameters={"X": TimeSeries},
    output_type=TimeSeries,
)
class TransformersTask(TaskBase):
    """Task for all transformer modules"""
