"""
A hypothetical inner block that transforms some output type rather than taking domain input
"""
# Local
from ...data_model.sample import SampleOutputType, SampleTask
import caikit.core


@caikit.core.block("00110203-baad-beef-0809-0a0b02dd0e0f", "SampleBlock", "0.0.1")
class InnerBlock(caikit.core.BlockBase):
    def __init__(self, batch_size=64, learning_rate=0.0015):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def run(self, some_input: SampleOutputType) -> SampleOutputType:
        return SampleOutputType(f"nested greeting: {some_input.greeting}")
