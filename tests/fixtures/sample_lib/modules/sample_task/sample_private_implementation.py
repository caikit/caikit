"""
A sample module for sample things!
"""
# Standard
import os

# Local
from ...data_model.sample import SampleInputTypePrivate, SampleOutputType, SampleTask
from .sample_implementation import SampleModule
import caikit.core


@caikit.core.module(
    "b38be66a-81f2-49e7-a767-f53072f24b8b", "SampleModulePrivate", "0.0.1", SampleTask
)
class SampleModulePrivate(SampleModule):
    """Simple wrapper around SampleModule that swaps the input type to the
    inherited private type
    """

    def run(
        self, sample_input: SampleInputTypePrivate, throw: bool = False
    ) -> SampleOutputType:
        return super().run(sample_input, throw)
