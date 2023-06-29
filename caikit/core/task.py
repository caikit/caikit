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
from typing import Callable, Dict, List, Type, TypeVar, Union
import collections
import dataclasses
import enum
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

_InferenceMethodBaseT = TypeVar("_InferenceMethodBaseT", bound=Callable)

_STREAM_OUT_ANNOTATION = "streaming_output_type"
_STREAM_PARAMS_ANNOTATION = "streaming_params"
_UNARY_OUT_ANNOTATION = "unary_output_type"
_UNARY_PARAMS_ANNOTATION = "unary_params"


class StreamingFlavor(enum.Enum):
    UNARY_UNARY = 1
    UNARY_STREAM = 2
    STREAM_UNARY = 3
    STREAM_STREAM = 4


class TaskBase:
    """The TaskBase defines the interface for an abstract AI task

    An AI task is a logical function signature which, when implemented, performs
    a task in some AI domain. The key property of a task is that the set of
    required input argument types and the output value type are consistent
    across all implementations of the task.
    """

    @dataclasses.dataclass
    class InferenceMethodPtr:
        method_name: str
        streaming_flavor: StreamingFlavor

    deferred_method_decorators: Dict[
        Type["TaskBase"], Dict[str, List["TaskBase.InferenceMethodPtr"]]
    ] = {}

    @classmethod
    def taskmethod(
        cls, streaming_flavor: StreamingFlavor = StreamingFlavor.UNARY_UNARY
    ) -> Callable[[_InferenceMethodBaseT], _InferenceMethodBaseT]:
        def decorator(inference_method: _InferenceMethodBaseT) -> _InferenceMethodBaseT:
            cls.deferred_method_decorators.setdefault(cls, {})
            fq_mod_name = ".".join(
                [
                    inference_method.__module__,
                    *inference_method.__qualname__.split(".")[0:-1],
                ]
            )
            cls.deferred_method_decorators[cls].setdefault(fq_mod_name, [])
            cls.deferred_method_decorators[cls][fq_mod_name].append(
                TaskBase.InferenceMethodPtr(
                    method_name=inference_method.__name__,
                    streaming_flavor=streaming_flavor,
                )
            )
            return inference_method

        return decorator

    @classmethod
    def deferred_method_decoration(cls, module: Type):
        if cls.has_inference_method_decorators(module):
            keyname = _make_keyname_for_module(module)
            deferred_decorations = cls.deferred_method_decorators[cls][keyname]
            for decoration in deferred_decorations:
                signature = CaikitMethodSignature(module, decoration.method_name)
                cls.validate_run_signature(signature, decoration.streaming_flavor)

    @classmethod
    def has_inference_method_decorators(cls, module_class: Type) -> bool:
        if cls not in cls.deferred_method_decorators:
            return False
        return (
            _make_keyname_for_module(module_class)
            in cls.deferred_method_decorators[cls]
        )

    @classmethod
    def validate_run_signature(
        cls, signature: CaikitMethodSignature, streaming_flavor: StreamingFlavor
    ) -> None:
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
            for parameter_name in cls.get_required_parameters(streaming_flavor)
            if parameter_name not in signature.parameters
        ]
        if missing_required_params:
            raise TypeError(
                f"Required parameters {missing_required_params} not in signature for module: "
                f"{signature.module}"
            )

        type_mismatch_errors = []
        for parameter_name, parameter_type in cls.get_required_parameters(
            streaming_flavor
        ).items():
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

        cls._raise_on_wrong_output_type(
            signature.return_type, signature.module, streaming_flavor
        )

    @classmethod
    def get_required_parameters(
        cls, flavor: StreamingFlavor
    ) -> Dict[str, ValidInputTypes]:
        """Get the set of input types required by this task"""
        if flavor in (StreamingFlavor.UNARY_UNARY, StreamingFlavor.UNARY_STREAM):
            if _UNARY_PARAMS_ANNOTATION not in cls.__annotations__:
                raise ValueError("No unary inputs are specified for this task")
            return cls.__annotations__[_UNARY_PARAMS_ANNOTATION]
        if _STREAM_PARAMS_ANNOTATION not in cls.__annotations__:
            raise ValueError("No streaming inputs are specified for this task")
        return cls.__annotations__[_STREAM_PARAMS_ANNOTATION]

    @classmethod
    def get_output_type(cls, flavor: StreamingFlavor) -> Type[DataBase]:
        """Get the output type for this task

        NOTE: This method is automatically configured by the @task decorator
            and should not be overwritten by child classes.
        """
        if flavor in (StreamingFlavor.UNARY_UNARY, StreamingFlavor.STREAM_UNARY):
            if _UNARY_OUT_ANNOTATION not in cls.__annotations__:
                raise ValueError("No unary outputs are specified for this task")
            return cls.__annotations__[_UNARY_OUT_ANNOTATION]
        if _STREAM_OUT_ANNOTATION not in cls.__annotations__:
            raise ValueError("No streaming outputs are specified for this task")
        return cls.__annotations__[_STREAM_OUT_ANNOTATION]

    @classmethod
    def is_output_streaming_task(cls) -> bool:
        """Returns true if this task has streaming output

        NOTE: This method is automatically configured by the @task decorator
            and should not be overwritten by child classes.
        """
        raise NotImplementedError("This is implemented by the @task decorator!")

    @classmethod
    def _raise_on_wrong_output_type(
        cls, output_type, module, streaming_flavor: StreamingFlavor
    ):
        task_output_type = cls.get_output_type(streaming_flavor)

        if cls._subclass_check(output_type, task_output_type):
            # Basic case, same type or subclass of it
            return

        if typing.get_origin(output_type) == Union:
            for union_type in typing.get_args(output_type):
                if cls._subclass_check(union_type, task_output_type):
                    # Something in the union has an acceptable type
                    return

        # Do some streaming checks
        if streaming_flavor in (
            StreamingFlavor.UNARY_STREAM,
            StreamingFlavor.STREAM_STREAM,
        ):
            if cls._is_iterable_type(output_type):
                # task_output_type is already guaranteed to be Iterable[T]
                streaming_type = typing.get_args(task_output_type)[0]

                for iterable_type in typing.get_args(output_type):
                    if cls._subclass_check(iterable_type, streaming_type):
                        return

        raise TypeError(
            f"Wrong output type for module {module}: "
            f"Found {output_type} but expected {task_output_type}"
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


def task(*_, **kwargs) -> Callable[[Type[TaskBase]], Type[TaskBase]]:
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

        error.subclass_check("<COR19436440E>", cls, TaskBase)

        # Backwards compatibility with old-style @tasks
        if (
            "required_parameters" in kwargs
            and _UNARY_PARAMS_ANNOTATION not in cls.__annotations__
        ):
            cls.__annotations__[_UNARY_PARAMS_ANNOTATION] = kwargs[
                "required_parameters"
            ]
        if "output_type" in kwargs and _UNARY_OUT_ANNOTATION not in cls.__annotations__:
            output_type = kwargs["output_type"]
            if cls._is_iterable_type(output_type):
                cls.__annotations__[_STREAM_OUT_ANNOTATION] = kwargs["output_type"]
            else:
                cls.__annotations__[_UNARY_OUT_ANNOTATION] = kwargs["output_type"]
        # End Backwards compatibility

        error.value_check(
            "<COR12671910E>",
            _UNARY_PARAMS_ANNOTATION in cls.__annotations__
            or _STREAM_PARAMS_ANNOTATION in cls.__annotations__,
            "At least one input type must be set on a task",
        )
        error.value_check(
            "<COR12671910E>",
            _UNARY_OUT_ANNOTATION in cls.__annotations__
            or _STREAM_OUT_ANNOTATION in cls.__annotations__,
            "At least one output type must be set on a task",
        )

        if _UNARY_OUT_ANNOTATION in cls.__annotations__:
            error.subclass_check(
                "<COR12766440E>", cls.__annotations__[_UNARY_OUT_ANNOTATION], DataBase
            )
        else:
            error.subclass_check(
                "<COR12766440E>",
                typing.get_origin(cls.__annotations__[_STREAM_OUT_ANNOTATION]),
                collections.abc.Iterable,
            )

        if _UNARY_PARAMS_ANNOTATION in cls.__annotations__:
            params_dict = cls.__annotations__[_UNARY_PARAMS_ANNOTATION]
            error.type_check("<COR19906440E>", dict, params_dict=params_dict)
            error.type_check_all(
                "<COR00123440E>", str, params_dict_keys=params_dict.keys()
            )
            # TODO: check proto-ability of things
        else:
            params_dict = cls.__annotations__[_STREAM_PARAMS_ANNOTATION]
            error.type_check("<COR19556230E>", dict, params_dict=params_dict)
            error.value_check(
                "<COR56569734E>",
                len(params_dict) == 1,
                "Only a single streaming input type supported",
            )
            error.type_check_all(
                "<COR58796465E>", str, params_dict_keys=params_dict.keys()
            )
            for v in params_dict.values():
                error.subclass_check(
                    "<COR52740295E>", typing.get_origin(v), collections.abc.Iterable
                )

        def is_output_streaming_task(_):
            return _STREAM_OUT_ANNOTATION in cls.__annotations__

        def is_input_streaming_task(_):
            return _STREAM_PARAMS_ANNOTATION in cls.__annotations__

        setattr(cls, "is_output_streaming_task", classmethod(is_output_streaming_task))
        setattr(cls, "is_input_streaming_task", classmethod(is_input_streaming_task))

        return cls

    return decorator


def _make_keyname_for_module(module_class: Type) -> str:
    return ".".join([module_class.__module__, module_class.__qualname__])
