"""
All timeseries tasks used by modules
"""

# Local
from .data_model.timeseries import TimeSeries
from .data_model.timeseries_evaluate import EvaluationResult
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
