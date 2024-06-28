"""
A bidi-streaming module for streaming things!

"""
# Standard
from typing import Iterable, Optional

# Local
from ...data_model.sample import (
    BidiStreamingTask,
    SampleInputType,
    SampleListInputType,
    SampleOutputType,
)
from caikit.core.data_model import DataStream
from caikit.core.modules import ModuleLoader, ModuleSaver
import caikit.core


@caikit.core.module(
    "00110203-0123-0456-0722-0a0b02dd0e0f", "SampleModule", "0.0.1", BidiStreamingTask
)
class BidiStreamingModule(caikit.core.ModuleBase):
    def __init__(self, stream_size=10):
        super().__init__()
        self.stream_size = stream_size

    @classmethod
    def load(cls, model_path, **kwargs):
        loader = ModuleLoader(model_path)
        config = loader.config
        return cls(config["stream_size"])

    @BidiStreamingTask.taskmethod(input_streaming=True, output_streaming=True)
    def run_bidi_stream(
        self, sample_inputs: DataStream[str]
    ) -> DataStream[SampleOutputType]:
        """
        Args:
            sample_inputs caikit.core.data_model.DataStream[str]: the input

        Returns:
            caikit.core.data_model.DataStream[sample_lib.data_model.SampleOutputType]: The output
                stream
        """
        sample_input = sample_inputs.peek()
        list_ = [
            SampleOutputType(f"Hello {sample_input}") for x in range(self.stream_size)
        ]
        stream = DataStream.from_iterable(list_)
        return stream

    def save(self, model_path):
        module_saver = ModuleSaver(
            self,
            model_path=model_path,
        )
        with module_saver:
            config_options = {"stream_size": self.stream_size}
            module_saver.update_config(config_options)
