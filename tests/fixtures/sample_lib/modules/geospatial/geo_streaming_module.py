"""
A sample module for sample things!
"""
# Standard
from typing import Iterable
import os

# Local
from ...data_model.sample import (
    GeoSpatialTask,
    SampleInputType,
    SampleOutputType,
    SampleTask,
    SampleTrainingType,
)
from caikit.core.data_model import DataStream
from caikit.core.modules import ModuleLoader, ModuleSaver
import caikit.core


@caikit.core.module(
    "e0e1e9b1-3cbb-411d-b066-3b8e1a19e46d",
    "WhyAreNamesStillHere",
    "0.0.1",
    GeoSpatialTask,
)
class GeoStreamingModule(caikit.core.ModuleBase):
    @GeoSpatialTask.taskmethod(input_streaming=True, output_streaming=True)
    def run_bidi_stream(
        self, lats: DataStream[float], lons: DataStream[float], name: str
    ) -> Iterable[SampleOutputType]:
        """
        Args:
            sample_inputs (caikit.core.data_model.DataStream[sample_lib.data_model.SampleInputType]): the input

        Returns:
            caikit.core.data_model.DataStream[sample_lib.data_model.SampleOutputType]: The output
                stream
        """
        for lat, lon in DataStream.zip(lats, lons):
            yield SampleOutputType(greeting=f"Hello from {name} at {lat}°, {lon}°")
