"""
A sample block for sample things!
"""
# Standard
from typing import List

# Local
from ...config import lib_config
from ...data_model.sample import SampleInputType, SampleOutputType, SampleTrainingType
from caikit.core.data_model import DataStream
from caikit.core.module import ModuleLoader, ModuleSaver
import caikit.core


@caikit.core.block("00af2203-0405-0607-0263-0a0b02dd0c2f", "ListBlock", "0.0.1")
class ListBlock(caikit.core.BlockBase):
    POISON_PILL_NAME = "Bob Marley"

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
        poison_pills: List[str] = [POISON_PILL_NAME],
    ) -> "ListBlock":
        """Sample training method that produces a trained model"""
        # Barf if we were incorrectly passed data not in datastream format
        assert isinstance(training_data, DataStream)
        assert len(poison_pills) > 0
        return cls(batch_size=batch_size)
