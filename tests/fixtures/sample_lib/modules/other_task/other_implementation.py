"""
A sample module for sample things!
"""
# Standard
from dataclasses import field
from typing import Dict, Union

# Local
from ...data_model.sample import OtherOutputType, OtherTask, SampleInputType
from caikit.core.data_model import DataStream
from caikit.core.data_model.json_dict import JsonDict
from caikit.core.modules import ModuleLoader, ModuleSaver
import caikit.core


@caikit.core.module(
    "33221100-0405-0607-0809-0a0b02dd0e0f", "OtherModule", "0.0.1", OtherTask
)
class OtherModule(caikit.core.ModuleBase):
    def __init__(self, batch_size=64, learning_rate=0.0015, training_parameters={}):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.training_parameters = training_parameters

    def run(
        self, sample_input: Union[SampleInputType, str]
    ) -> Union[OtherOutputType, str]:
        return OtherOutputType(
            f"goodbye: {sample_input.name} {self.batch_size} times {self.training_parameters.get('layer_sizes')}"
        )

    @classmethod
    def load(cls, model_path, **kwargs):
        loader = ModuleLoader(model_path)
        config = loader.config
        return cls(
            config["train"]["batch_size"],
            config["train"]["learning_rate"],
            config["train"]["training_parameters"],
        )

    def save(self, model_path):
        module_saver = ModuleSaver(
            self,
            model_path=model_path,
        )
        with module_saver:
            config_options = {
                "train": {
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "training_parameters": self.training_parameters,
                },
            }

            module_saver.update_config(config_options)

    @classmethod
    def train(
        cls,
        training_data: DataStream[int],
        sample_input: Union[SampleInputType, str],
        batch_size: int = 64,
        training_parameters_json_dict: JsonDict = None,
        training_parameters: Dict[str, int] = field(default_factory=dict),
    ) -> "OtherModule":
        """Sample training method that produces a trained model"""
        assert type(sample_input) == SampleInputType or str
        # Barf if we were incorrectly passed data not in datastream format
        assert isinstance(training_data, DataStream)
        assert training_parameters_json_dict is not None
        assert training_parameters is not None
        assert batch_size > 0
        return cls(batch_size=batch_size, training_parameters=training_parameters)
