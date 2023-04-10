"""
A sample block for sample things!
"""
# Local
from ...config import lib_config
from ...data_model.sample import OtherOutputType, SampleInputType, SampleTrainingType
from caikit.core.data_model import DataStream
from caikit.core.module import ModuleLoader, ModuleSaver
import caikit.core


@caikit.core.block("33221100-0405-0607-0809-0a0b02dd0e0f", "OtherBlock", "0.0.1")
class OtherBlock(caikit.core.BlockBase):
    def __init__(self, batch_size=64, learning_rate=0.0015):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def run(self, sample_input: SampleInputType) -> OtherOutputType:
        return OtherOutputType(f"goodbye: {sample_input.name} {self.batch_size} times")

    @classmethod
    def load(cls, model_path, **kwargs):
        loader = ModuleLoader(model_path)
        config = loader.config
        return cls(config["train"]["batch_size"], config["train"]["learning_rate"])

    def save(self, model_path):
        module_saver = ModuleSaver(
            self,
            model_path=model_path,
            library_name="sample_lib",
            library_version=lib_config.library_version,
        )
        with module_saver:
            config_options = {
                "train": {
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                },
            }

            module_saver.update_config(config_options)

    @classmethod
    def train(
        cls, training_data: DataStream[int], batch_size: int = 64
    ) -> "OtherBlock":
        """Sample training method that produces a trained model"""
        # Barf if we were incorrectly passed data not in datastream format
        assert type(training_data) == DataStream
        assert batch_size > 0
        return cls(batch_size=batch_size)
