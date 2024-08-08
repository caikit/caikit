"""
A sample module for sample things!
"""
# Standard
from typing import Dict, Iterable, List, Optional, Union
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
from caikit.core.data_model.runtime_context import RuntimeServerContextType
from caikit.core.exceptions.caikit_core_exception import (
    CaikitCoreException,
    CaikitCoreStatusCode,
)
from caikit.core.modules import ModuleLoader, ModuleSaver
from caikit.runtime import trace
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
        self.stream_size: int = stream_size
        # Used for failing the first number of requests
        self.request_attempt_tracker: Dict[str, int] = {}
        self._tracer = trace.get_tracer(__name__)

    @classmethod
    def load(cls, model_path, **kwargs):
        loader = ModuleLoader(model_path)
        config = loader.config
        return cls(config["train"]["batch_size"], config["train"]["learning_rate"])

    @SampleTask.taskmethod(context_arg="context")
    def run(
        self,
        sample_input: SampleInputType,
        throw: bool = False,
        error: Optional[str] = None,
        request_id: Optional[str] = None,
        throw_first_num_requests: Optional[int] = None,
        sleep_time: float = 0,
        sleep_increment: float = 0.001,
        context: Optional[RuntimeServerContextType] = None,
    ) -> SampleOutputType:
        """
        Args:
            sample_input (SampleInputType): the input
            throw (bool, optional): If this request should throw an error. Defaults to False.
            error (Optional[str], optional): The error string to throw. Defaults to None.
            request_id (Optional[str], optional): The request id for tracking the end-user identity
                for throw_first_num_requests. Defaults to None.
            throw_first_num_requests (Optional[int], optional): How many requests to throw an error
                for before being successful. Defaults to None.
            sleep_time float: How long to sleep before returning a result. Defaults to 0.
            sleep_increment float: How large of increments to sleep in.
            context (Optional[RuntimeServerContextType]): The context for the runtime server request
        Returns:
            SampleOutputType: The output
        """
        span_name = f"{__name__}.{type(self).__name__}.run"
        with trace.start_child_span(context, span_name):
            if sleep_time:
                start_time = time.time()
                while time.time() - sleep_time < start_time:
                    time.sleep(sleep_increment)

            if throw:
                self._raise_error(error)

            if throw_first_num_requests and not request_id:
                self._raise_error(
                    "throw_first_num_requests requires providing a request_id"
                )

            # If a throw_first_num_requests was provided  then increment the tracker and raise an exception
            # until the number of requests is high enough
            if throw_first_num_requests:
                self.request_attempt_tracker[request_id] = (
                    self.request_attempt_tracker.get(request_id, 0) + 1
                )
                if self.request_attempt_tracker[request_id] <= throw_first_num_requests:
                    self._raise_error(error)

            assert isinstance(sample_input, SampleInputType)
            if sample_input.name == self.POISON_PILL_NAME:
                raise ValueError(f"{self.POISON_PILL_NAME} is not allowed!")
            return SampleOutputType(f"Hello {sample_input.name}")

    @SampleTask.taskmethod(output_streaming=True)
    def run_stream_out(
        self,
        sample_input: SampleInputType,
        err_stream: bool = False,
        error: Optional[str] = None,
    ) -> DataStream[SampleOutputType]:
        """
        Args:
            sample_input (sample_lib.data_model.SampleInputType): the input
            err_stream (bool, optional): An optional parameter to error out the stream
            error (Optional[str], optional): The error string to error out. Defaults to None.

        Returns:
            caikit.core.data_model.DataStream[sample_lib.data_model.SampleOutputType]: The output
                stream
        """
        list_ = [
            SampleOutputType(f"Hello {sample_input.name} stream")
            for x in range(self.stream_size)
        ]
        # raise errors when the stream is iterated, not before.
        def raise_exception(error):
            if error:
                self._raise_error(error)
            else:
                raise ValueError("raising a ValueError")

        stream = (
            DataStream.from_iterable(list_)
            if not err_stream
            else DataStream.from_iterable([1]).map(lambda x: raise_exception(error))
        )
        return stream

    @SampleTask.taskmethod(input_streaming=True)
    def run_stream_in(
        self,
        sample_inputs: DataStream[SampleInputType],
        greeting: str = "Hello Friends",
    ) -> SampleOutputType:
        """
        Args:
            sample_inputs (caikit.core.data_model.DataStream[sample_lib.data_model.SampleInputType]): the input
            greeting (str): Greeting to use for the response
        Returns:
            sample_lib.data_model.SampleOutputType]: The combination of inputs
                stream
        """
        return SampleOutputType(
            greeting=f"{greeting}{','.join([val.name for val in sample_inputs])}"
        )

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

    def _raise_error(self, error: str):
        if error:
            if error == "GRPC_RESOURCE_EXHAUSTED":
                raise _channel._InactiveRpcError(
                    _channel._RPCState(
                        due=(),
                        details="Model is overloaded",
                        initial_metadata=None,
                        trailing_metadata=None,
                        code=StatusCode.RESOURCE_EXHAUSTED,
                    ),
                )
            elif error == "CORE_EXCEPTION":
                raise CaikitCoreException(
                    status_code=CaikitCoreStatusCode.INVALID_ARGUMENT,
                    message="invalid argument",
                )
        raise RuntimeError(error)
