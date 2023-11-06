"""
A sample module for sample things!
"""
# Standard
from typing import Iterable, List, Optional, Union
import os
import time

# Third Party
from grpc import StatusCode, _channel

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

    def __init__(self, batch_size=64, learning_rate=0.0015, stream_size=10):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.stream_size = stream_size

    @classmethod
    def load(cls, model_path, **kwargs):
        loader = ModuleLoader(model_path)
        config = loader.config
        return cls(config["train"]["batch_size"], config["train"]["learning_rate"])

    @SampleTask.taskmethod()
    def run(
        self,
        sample_input: SampleInputType,
        throw: bool = False,
        error: Optional[str] = None,
    ) -> SampleOutputType:
        """
        Args:
            sample_input (sample_lib.data_model.SampleInputType): the input

        Returns:
            sample_lib.data_model.SampleOutputType: The output
        """
        if throw:
            if error and error == "GRPC_RESOURCE_EXHAUSTED":
                raise _channel._InactiveRpcError(
                    _channel._RPCState(
                        due=(),
                        details="Model is overloaded",
                        initial_metadata=None,
                        trailing_metadata=None,
                        code=StatusCode.RESOURCE_EXHAUSTED,
                    ),
                )
            raise RuntimeError("barf!")
        assert isinstance(sample_input, SampleInputType)
        if sample_input.name == self.POISON_PILL_NAME:
            raise ValueError(f"{self.POISON_PILL_NAME} is not allowed!")
        return SampleOutputType(f"Hello {sample_input.name}")

    @SampleTask.taskmethod(output_streaming=True)
    def run_stream_out(
        self, sample_input: SampleInputType, err_stream: bool = False
    ) -> DataStream[SampleOutputType]:
        """
        Args:
            sample_input (sample_lib.data_model.SampleInputType): the input
            err_stream (bool): An optional parameter to error out the stream

        Returns:
            caikit.core.data_model.DataStream[sample_lib.data_model.SampleOutputType]: The output
                stream
        """
        list_ = [
            SampleOutputType(f"Hello {sample_input.name} stream")
            for x in range(self.stream_size)
        ]
        # raise a value error when the stream is iterated, not before.
        def raise_exception():
            raise ValueError("raising a ValueError")

        stream = (
            DataStream.from_iterable(list_)
            if not err_stream
            else DataStream.from_iterable([1]).map(lambda x: raise_exception())
        )
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

    def save(self, model_path, **kwargs):
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
            if "keyword" in kwargs:
                config_options["test_keyword"] = True

            module_saver.update_config(config_options)

    @classmethod
    def train(
        cls,
        training_data: DataStream[SampleTrainingType],
        union_list: Optional[Union[List[str], List[int]]] = None,
        batch_size: int = 64,
        oom_exit: bool = False,
        oom_exit_code: Optional[int] = None,
        sleep_time: float = 0,
        sleep_increment: float = 0.001,
        **kwargs,
    ) -> "SampleModule":
        """Sample training method that produces a trained model"""

        # If requested, set an event to indicate that the training has started
        start_event = kwargs.get("start_event")
        if start_event is not None:
            start_event.set()

        # If needed, wait for an event
        # NOTE: We need to pull this from **kwargs because it's not a valid arg
        #   for train API deduction. It's only needed for testing purposes, so
        #   this is definitely a non-standard usage pattern!
        wait_event = kwargs.get("wait_event")
        if wait_event is not None:
            # Set a timeout here so tests don't hang forever in failure states
            wait_event.wait(timeout=1)

        # If needed, wait for a long time
        # NOTE: DestroyableThread is a "best effort" at destroying a threaded
        #   work and is not actually capable of destroying many type of work. If
        #   this is written as time.sleep(sleep_time), the interrupt will not
        #   work.
        if sleep_time:
            start_time = time.time()
            while time.time() - sleep_time < start_time:
                time.sleep(sleep_increment)

        if oom_exit:
            # exit with OOM code. Note _exit method is used to exit the
            # process with specified status without calling cleanup handlers
            # to replicate OOM scenario
            if oom_exit_code:
                os._exit(oom_exit_code)
            else:
                os._exit(137)

        if batch_size == cls.POISON_PILL_BATCH_SIZE:
            raise ValueError(
                f"Batch size of {cls.POISON_PILL_BATCH_SIZE} is not allowed!"
            )
        # Barf if we were incorrectly passed data not in datastream format
        assert isinstance(training_data, DataStream)
        for _ in training_data:
            # Consume the stream
            pass
        if union_list:
            assert isinstance(union_list, List)
            assert len(union_list) > 0
        return cls(batch_size=batch_size)
