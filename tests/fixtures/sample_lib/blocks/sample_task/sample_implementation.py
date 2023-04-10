"""
A sample block for sample things!
"""
# Local
from ...config import lib_config
from ...data_model.sample import SampleInputType, SampleOutputType, SampleTrainingType
from caikit.core.data_model import DataStream
from caikit.core.module import ModuleLoader, ModuleSaver
import caikit.core


@caikit.core.block("00110203-0405-0607-0809-0a0b02dd0e0f", "SampleBlock", "0.0.1")
class SampleBlock(caikit.core.BlockBase):
    POISON_PILL_NAME = "Bob Marley"
    POISON_PILL_BATCH_SIZE = 999

    def __init__(self, batch_size=64, learning_rate=0.0015):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    @classmethod
    def load(cls, model_path, **kwargs):
        loader = ModuleLoader(model_path)
        config = loader.config
        return cls(config["train"]["batch_size"], config["train"]["learning_rate"])

    def run(
        self, sample_input: SampleInputType, throw: bool = False
    ) -> SampleOutputType:
        """
        Args:
            sample_input (sample_lib.data_model.SampleInputType): the input

        Returns:
            sample_lib.data_model.SampleOutputType: The output
        """
        if throw:
            raise RuntimeError("barf!")
        if sample_input.name == self.POISON_PILL_NAME:
            raise ValueError(f"{self.POISON_PILL_NAME} is not allowed!")
        return SampleOutputType(f"Hello {sample_input.name}")

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
        cls,
        training_data: DataStream[SampleTrainingType],
        batch_size: int = 64,
    ) -> "SampleBlock":
        """Sample training method that produces a trained model"""
        if batch_size == cls.POISON_PILL_BATCH_SIZE:
            raise ValueError(
                f"Batch size of {cls.POISON_PILL_BATCH_SIZE} is not allowed!"
            )
        # Barf if we were incorrectly passed data not in datastream format
        assert isinstance(training_data, DataStream)
        return cls(batch_size=batch_size)
