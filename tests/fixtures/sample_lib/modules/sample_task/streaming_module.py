"""
A streaming module for streaming things!

NB: This is here for backwards-compatibility support with ye-olde basic streaming-out tasks
"""

# Local
from ...data_model.sample import SampleInputType, SampleOutputType, StreamingTask
from caikit.core.data_model import DataStream
from caikit.core.modules import ModuleLoader, ModuleSaver
import caikit.core


@caikit.core.module(
    "00110203-0123-0456-0789-0a0b02dd0e0f", "SampleModule", "0.0.1", StreamingTask
)
class StreamingModule(caikit.core.ModuleBase):
    def __init__(self, stream_size=10):
        super().__init__()
        self.stream_size = stream_size

    @classmethod
    def load(cls, model_path, **kwargs):
        loader = ModuleLoader(model_path)
        config = loader.config
        return cls(config["stream_size"])

    def run(self, sample_input: SampleInputType) -> DataStream[SampleOutputType]:
        """
        Args:
            sample_input (sample_lib.data_model.SampleInputType): the input

        Returns:
            caikit.core.data_model.DataStream[sample_lib.data_model.SampleOutputType]: The output
                stream
        """
        list_ = [
            SampleOutputType(f"Hello {sample_input.name}")
            for x in range(self.stream_size)
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
