"""
A sample module for sample things!
"""
# Standard
from typing import Iterable, List, Union
import os

# Local
from ...data_model.sample import (
    SampleInputType,
    SampleOutputType,
    SampleTask,
    SampleTrainingType,
)
from caikit.core.data_model import DataStream
from caikit.core.modules import ModuleLoader, ModuleSaver
import caikit.core


@caikit.core.module(
    "00110203-0405-0607-0809-0a0b02dd0e0f", "SampleModule", "0.0.1", SampleTask
)
class SampleModule(caikit.core.ModuleBase):
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

    @SampleTask.taskmethod()
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

    @SampleTask.taskmethod(output_streaming=True)
    def run_stream_out(
        self, sample_input: SampleInputType
    ) -> DataStream[SampleOutputType]:
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

    @SampleTask.taskmethod(input_streaming=True, output_streaming=True)
    def run_bidi_stream(
        self, sample_inputs: DataStream[SampleInputType]
    ) -> Iterable[SampleOutputType]:
        """
        Args:
            sample_inputs (caikit.core.data_model.DataStream[sample_lib.data_model.SampleInputType]): the input

        Returns:
            caikit.core.data_model.DataStream[sample_lib.data_model.SampleOutputType]: The output
                stream
        """
        for sample_input in sample_inputs:
            yield self.run(sample_input)

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
                },
            }

            module_saver.update_config(config_options)

    @classmethod
    def train(
        cls,
        training_data: DataStream[SampleTrainingType],
        union_list: Union[List[str], List[int]] = None,
        batch_size: int = 64,
        oom_exit: bool = False,
    ) -> "SampleModule":
        """Sample training method that produces a trained model"""

        if oom_exit:
            # exit with OOM code. Note _exit method is used to exit the
            # process with specified status without calling cleanup handlers
            # to replicate OOM scenario
            os._exit(137)

        if batch_size == cls.POISON_PILL_BATCH_SIZE:
            raise ValueError(
                f"Batch size of {cls.POISON_PILL_BATCH_SIZE} is not allowed!"
            )
        # Barf if we were incorrectly passed data not in datastream format
        assert isinstance(training_data, DataStream)
        if union_list:
            assert isinstance(union_list.union_list.values, List)
            assert len(union_list.union_list.values) > 0
        return cls(batch_size=batch_size)
