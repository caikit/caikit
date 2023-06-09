"""
A hypothetical inner module that transforms some output type rather than taking domain input
"""
# Local
from ...data_model.sample import SampleOutputType
from caikit.core import ModuleSaver
import caikit.core


@caikit.core.module("00110203-baad-beef-0809-0a0b02dd0e0f", "InnerModule", "0.0.1")
class InnerModule(caikit.core.ModuleBase):
    def __init__(self, batch_size=64, learning_rate=0.0015):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def run(self, some_input: SampleOutputType) -> SampleOutputType:
        return SampleOutputType(f"nested greeting: {some_input.greeting}")

    def save(self, model_path):
        module_saver = ModuleSaver(
            self,
            model_path=model_path,
        )
        with module_saver:
            pass

    @classmethod
    def load(cls, model_path):
        return cls()
