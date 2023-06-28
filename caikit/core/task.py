# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Standard
from inspect import isclass
from typing import Callable, Dict, List, Type, Union
import collections
import typing

# First Party
from alog import alog

# Local
from .data_model import DataStream
from .data_model.base import DataBase
from .signature_parsing import CaikitMethodSignature
from .toolkit.errors import error_handler

log = alog.use_channel("TASK_BASE")
error = error_handler.get(log)

ProtoableInputTypes = Type[Union[int, float, str, bytes, bool, DataBase]]
ValidInputTypes = Union[
    ProtoableInputTypes, List[ProtoableInputTypes], DataStream[ProtoableInputTypes]
]


class TaskBase:
    """The TaskBase defines the interface for an abstract AI task

    An AI task is a logical function signature which, when implemented, performs
    a task in some AI domain. The key property of a task is that the set of
    required input argument types and the output value type are consistent
    across all implementations of the task.
    """

    @classmethod
    def validate_run_signature(cls, signature: CaikitMethodSignature) -> None:
        #
        if not signature.parameters:
            raise ValueError(
                "Task could not be validated, no .run parameters were provided"
            )
        if signature.return_type is None:
            raise ValueError(
                "Task could not be validated, no .run return type was provided"
            )

        missing_required_params = [
            parameter_name
            for parameter_name in cls.get_required_parameters()
            if parameter_name not in signature.parameters
        ]
        if missing_required_params:
            raise TypeError(
                f"Required parameters {missing_required_params} not in signature for module: "
                f"{signature.module}"
            )

        type_mismatch_errors = []
        for parameter_name, parameter_type in cls.get_required_parameters().items():
            signature_type = signature.parameters[parameter_name]
            if parameter_type != signature_type:
                if typing.get_origin(
                    signature_type
                ) == typing.Union and parameter_type in typing.get_args(signature_type):
                    continue
                type_mismatch_errors.append(
                    f"Parameter {parameter_name} has type {signature_type} but type \
                        {parameter_type} is required"
                )
        if type_mismatch_errors:
            raise TypeError(
                f"Wrong types provided for parameters to {signature.module}: {type_mismatch_errors}"
            )

        cls._raise_on_wrong_output_type(signature.return_type, signature.module)

    @classmethod
    def get_required_parameters(cls) -> Dict[str, ValidInputTypes]:
        """Get the set of input types required by this task

        NOTE: This method is automatically configured by the @task decorator
            and should not be overwritten by child classes.
        """
        raise NotImplementedError("This is implemented by the @task decorator!")

    @classmethod
    def get_output_type(cls) -> Type[DataBase]:
        """Get the output type for this task

        NOTE: This method is automatically configured by the @task decorator
            and should not be overwritten by child classes.
        """
        raise NotImplementedError("This is implemented by the @task decorator!")

    @classmethod
    def is_output_streaming_task(cls) -> bool:
        """Returns true if this task has streaming output

        NOTE: This method is automatically configured by the @task decorator
            and should not be overwritten by child classes.
        """
        raise NotImplementedError("This is implemented by the @task decorator!")

    @classmethod
    def _raise_on_wrong_output_type(cls, output_type, module):
        if cls._subclass_check(output_type, cls.get_output_type()):
            # Basic case, same type or subclass of it
            return

        if typing.get_origin(output_type) == Union:
            for union_type in typing.get_args(output_type):
                if cls._subclass_check(union_type, cls.get_output_type()):
                    # Something in the union has an acceptable type
                    return

        # Iterable output checks:
        # `output_type` here can be any iterable, so long as it is type annotated with the type
        # that is iterated over. So we first only check that it is iterable at all, and then look
        # at the annotated type's args.
        if cls._is_iterable_type(output_type) and cls.is_output_streaming_task():
            # In this case, the task decorator has already validated that the task has output type
            # Iterable[T] with exactly one T, so this is safe.
            streaming_type = typing.get_args(cls.get_output_type())[0]

            for iterable_type in typing.get_args(output_type):
                if cls._subclass_check(iterable_type, streaming_type):
                    return

        raise TypeError(
            f"Wrong output type for module {module}: "
            f"Found {output_type} but expected {cls.get_output_type()}"
        )

    @staticmethod
    def _subclass_check(this_type, that_type):
        """Wrapper around issubclass that first checks if both args are classes.
        Returns True if the types are the same, or they are both classes and this_type
        is a subclass of that_type
        """
        if this_type == that_type:
            return True
        if isclass(this_type) and isclass(that_type):
            return issubclass(this_type, that_type)
        return False

    @staticmethod
    def _is_iterable_type(typ: Type) -> bool:
        """Returns True if typ is an iterable type.
        Does not work for types like `list`, `tuple`, but we're interested here in `List[T]` etc.

        This is implemented this way to support older python versions where
        isinstance(typ, typing.Iterable) does not work
        """
        try:
            iter(typ)
            return True
        except TypeError:
            return False


def task(*args, **kwargs) -> Callable[[Type[TaskBase]], Type[TaskBase]]:
    """The decorator for AI Task classes.

    This defines an output data model type for the task, and a minimal set of required inputs
    that all public models implementing this task must accept.

    As an example, the `caikit.interfaces.nlp.SentimentTask` might look like::

        @task(
            required_inputs={
                "raw_document": caikit.interfaces.nlp.RawDocument
            },
            output_type=caikit.interfaces.nlp.SentimentPrediction
        )
        class SentimentTask(caikit.TaskBase):
            pass

    and a public model that implements this task might have a .run function that looks like::

        def run(raw_document: caikit.interfaces.nlp.RawDocument,
                inference_mode: str = "fast",
                device: caikit.interfaces.common.HardwareEnum) ->
                    caikit.interfaces.nlp.SentimentPrediction:
            # impl

    Note the run function may include other arguments beyond the minimal required inputs for
    the task.

    Args:
        required_parameters (Dict[str, ValidInputTypes]): The required parameters that all public
            models' .run functions must contain. A dictionary of parameter name to parameter
            type, where the types can be in the set of: - Python primitives
                - Caikit data models
                - Iterable containers of the above
                - Caikit model references (maybe?)
        output_type (Type[DataBase]): The output type of the task, which all public models'
            .run functions must return. This must be a caikit data model type.

    Returns:
        A decorator function for the task class, registering it with caikit's core registry of
            tasks.
    """
    # TODO: type checking on required_inputs
    # Check that return type is a data model or iterable of a data model
    # We explicitly require `Iterable[T]` on output type annotations
    # if typing.get_origin(output_type) == collections.abc.Iterable:
    #     error.value_check(
    #         "<COR12569910E>",
    #         len(typing.get_args(output_type)) == 1,
    #         "A single type T must be provided for tasks with output type Iterable[T].",
    #     )
    #     error.subclass_check(
    #         "<COR12766440E>", typing.get_args(output_type)[0], DataBase
    #     )
    #     output_streaming = True
    # else:
    #     error.subclass_check("<COR12766440E>", output_type, DataBase)
    #     output_streaming = False

    # def get_required_parameters(_):
    #     return cls.unary_params

    # def get_output_type(_):
    #     return output_type

    def decorator(cls: Type[TaskBase]) -> Type[TaskBase]:
        # get_required_parameters.__doc__ = f"""
        # Returns the set of input parameters required in the `run` function for any module that
        # implements the '{cls.__name__}' Task.

        # ({cls.__annotations__.unary_params})

        # Returns: Dict[str, Type]
        #     The parameter dictionary for the {cls.__name__} inference task
        # """

        # get_output_type.__doc__ = f"""
        # Returns the output type required of the `run` function in any module that implements the
        # {cls.__name__} task.

        # ({cls.__annotations__.unary_output_type})

        # Returns: Type
        #     The output type of the {cls.__name__} inference task
        # """
        # output_type = cls.__annotations__.unary_output_type
        # # setattr(cls, "get_required_parameters", classmethod(get_required_parameters))
        # # setattr(cls, "get_output_type", classmethod(get_output_type))

        # if typing.get_origin(output_type) == collections.abc.Iterable:
        #     error.value_check(
        #         "<COR12569910E>",
        #         len(typing.get_args(output_type)) == 1,
        #         "A single type T must be provided for tasks with output type Iterable[T].",
        #     )
        #     error.subclass_check(
        #         "<COR12766440E>", typing.get_args(output_type)[0], DataBase
        #     )
        #     output_streaming = True
        # else:
        #     error.subclass_check("<COR12766440E>", output_type, DataBase)
        #     output_streaming = False

        def is_output_streaming_task(_):
            return "streaming_output_type" in cls.__annotations__

        def is_input_streaming_task(_):
            return "streaming_params" in cls.__annotations__

        error.subclass_check("<COR19436440E>", cls, TaskBase)
        setattr(cls, "is_output_streaming_task", classmethod(is_output_streaming_task))
        setattr(cls, "is_input_streaming_task", classmethod(is_input_streaming_task))

        return cls

    return decorator
