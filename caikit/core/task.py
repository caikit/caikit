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
from typing import Callable, Dict, Iterable, List, Type, TypeVar, Union
import collections
import dataclasses
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

_STREAM_OUT_ANNOTATION = "__streaming_output_type"
_STREAM_PARAMS_ANNOTATION = "__streaming_params"
_UNARY_OUT_ANNOTATION = "__unary_output_type"
_UNARY_PARAMS_ANNOTATION = "__unary_params"


class TaskBase:
    """The TaskBase defines the interface for an abstract AI task

    An AI task is a logical function signature which, when implemented, performs
    a task in some AI domain. The key property of a task is that the set of
    required input argument types and the output value type are consistent
    across all implementations of the task.
    """

    @dataclasses.dataclass
    class InferenceMethodPtr:
        """Little container class that holds a method name and its flavor of streaming.
        i.e. the args to a `@TaskClass.taskmethod` decoration.
        """

        method_name: str  # the simple name of a method, like "run"
        input_streaming: bool
        output_streaming: bool

    deferred_method_decorators: Dict[
        Type["TaskBase"], Dict[str, List["TaskBase.InferenceMethodPtr"]]
    ] = {}

    @classmethod
    def taskmethod(
        cls, input_streaming: bool = False, output_streaming: bool = False
    ) -> Callable[[_InferenceMethodBaseT], _InferenceMethodBaseT]:
        """Decorates a module instancemethod and indicates whether the inputs and outputs should
        be handled as streams. This will trigger validation that the signature of this method
        is compatible with the task's definition of input and output types.

        The actual handling of validating the method and registering it is deferred until after
        the module class is created, which happens outside the context of this decoration.
        """

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
                    input_streaming=input_streaming,
                    output_streaming=output_streaming,
                )
            )
            return inference_method

        return decorator

    @classmethod
    def deferred_method_decoration(cls, module: Type):
        """Runs the actual decoration logic that `taskmethod` would have run if the module class
        existed during its lifetime.

        Validates that all decorated methods match the task's API expectations, and stores the
        signatures on the module class for access later.
        """
        if cls.has_inference_method_decorators(module):
            keyname = _make_keyname_for_module(module)
            deferred_decorations = cls.deferred_method_decorators[cls][keyname]
            for decoration in deferred_decorations:
                signature = CaikitMethodSignature(module, decoration.method_name)
                cls.validate_run_signature(
                    signature, decoration.input_streaming, decoration.output_streaming
                )

                module._INFERENCE_SIGNATURES.append(
                    (decoration.input_streaming, decoration.output_streaming, signature)
                )

    @classmethod
    def has_inference_method_decorators(cls, module_class: Type) -> bool:
        """Utility that returns true iff a module has any `@TaskClass.taskmethod` decorations"""
        if cls not in cls.deferred_method_decorators:
            return False
        return (
            _make_keyname_for_module(module_class)
            in cls.deferred_method_decorators[cls]
        )

    @classmethod
    def validate_run_signature(
        cls,
        signature: CaikitMethodSignature,
        input_streaming: bool,
        output_streaming: bool,
    ) -> None:
        """Validates that the provided method signature meets the api constraints defined in this
        task, for the given streaming flavors.

        Raises:
            ValueError if no type annotations were provided on the method
            TypeError if the type annotations do not meet the task's api constraints
        """
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
            for parameter_name in cls.get_required_parameters(input_streaming)
            if parameter_name not in signature.parameters
        ]
        if missing_required_params:
            raise TypeError(
                f"Required parameters {missing_required_params} not in signature for module: "
                f"{signature.module}"
            )

        type_mismatch_errors = []
        for parameter_name, parameter_type in cls.get_required_parameters(
            input_streaming
        ).items():
            signature_type = signature.parameters[parameter_name]
            if parameter_type != signature_type:
                if typing.get_origin(
                    signature_type
                ) == typing.Union and parameter_type in typing.get_args(signature_type):
                    continue
                if input_streaming and cls._is_iterable_type(parameter_type):
                    streaming_type = typing.get_args(parameter_type)[0]

                    for iterable_type in typing.get_args(signature_type):
                        if not cls._subclass_check(iterable_type, streaming_type):
                            raise TypeError(
                                f"Wrong input type for {parameter_name}, expected {parameter_type} \
                                  but got {signature_type}"
                            )
                else:
                    type_mismatch_errors.append(
                        f"Parameter {parameter_name} has type {signature_type} but type \
                            {parameter_type} is required"
                    )
        if type_mismatch_errors:
            raise TypeError(
                f"Wrong types provided for parameters to {signature.module}: {type_mismatch_errors}"
            )

        cls._raise_on_wrong_output_type(
            signature.return_type, signature.module, output_streaming
        )

    @classmethod
    def get_required_parameters(
        cls, input_streaming: bool
    ) -> Dict[str, Union[ValidInputTypes, Type[Iterable[ValidInputTypes]]]]:
        """Get the set of input types required by this task"""
        if not input_streaming:
            if _UNARY_PARAMS_ANNOTATION not in cls.__annotations__:
                raise ValueError("No unary inputs are specified for this task")
            return cls.__annotations__[_UNARY_PARAMS_ANNOTATION]
        if _STREAM_PARAMS_ANNOTATION not in cls.__annotations__:
            raise ValueError("No streaming inputs are specified for this task")
        return cls.__annotations__[_STREAM_PARAMS_ANNOTATION]

    @classmethod
    def get_output_type(cls, output_streaming: bool) -> Type[DataBase]:
        """Get the output type for this task

        NOTE: This method is automatically configured by the @task decorator
            and should not be overwritten by child classes.
        """
        if not output_streaming:
            if _UNARY_OUT_ANNOTATION not in cls.__annotations__:
                raise ValueError("No unary outputs are specified for this task")
            return cls.__annotations__[_UNARY_OUT_ANNOTATION]
        if _STREAM_OUT_ANNOTATION not in cls.__annotations__:
            raise ValueError("No streaming outputs are specified for this task")
        return cls.__annotations__[_STREAM_OUT_ANNOTATION]

    @classmethod
    def _raise_on_wrong_output_type(cls, output_type, module, output_streaming: bool):
        task_output_type = cls.get_output_type(output_streaming)

        if cls._subclass_check(output_type, task_output_type):
            # Basic case, same type or subclass of it
            return

        if typing.get_origin(output_type) == Union:
            for union_type in typing.get_args(output_type):
                if cls._subclass_check(union_type, task_output_type):
                    # Something in the union has an acceptable type
                    return

        # Do some streaming checks
        if output_streaming:
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


def task(
    unary_parameters: Dict[str, ValidInputTypes] = None,
    streaming_parameters: Dict[str, Type[Iterable[ValidInputTypes]]] = None,
    unary_output_type: Type[DataBase] = None,
    streaming_output_type: Type[Iterable[Type[DataBase]]] = None,
    **kwargs,
) -> Callable[[Type[TaskBase]], Type[TaskBase]]:
    """The decorator for AI Task classes.

    This defines an output data model type for the task, and a minimal set of required inputs
    that all public models implementing this task must accept.

    As an example, the `caikit.interfaces.nlp.SentimentTask` might look like::

        @task(
            unary_parameters={
                "raw_document": caikit.interfaces.nlp.RawDocument
            },
            streaming_parameters={
                "raw_documents": Iterable[caikit.interfaces.nlp.RawDocument]
            }
            unary_output_type=caikit.interfaces.nlp.SentimentPrediction
            streaming_output_type=Iterable[caikit.interfaces.nlp.SentimentPrediction]
        )
        class SentimentTask(caikit.TaskBase):
            pass

    and a module that implements this task might have methods like::

        @module(id="b9d98408-84c2-488c-8385-9d698effe60b", task=SentimentTask)
        class MyModule(ModuleBase):

            @SentimentTask.taskmethod()
            def run(raw_document: caikit.interfaces.nlp.RawDocument,
                    inference_mode: str = "fast") ->
                        caikit.interfaces.nlp.SentimentPrediction:
                # impl

            @SentimentTask.taskmethod(input_streaming=True, output_streaming=True)
            def run_bidi_stream(raw_documents: DataStream[caikit.interfaces.nlp.RawDocument])
                    -> DataStream[caikit.interfaces.nlp.SentimentPrediction]:
                # impl

    Note the run function may include other arguments beyond the minimal required inputs for
    the task.

    Args:
        unary_parameters (Dict[str, ValidInputTypes]): The required parameters that all module's
            unary-input inference methods must contain. A dictionary of parameter name to parameter
            type, where the types can be in the set of:
                - Python primitives
                - Caikit data models
                - Iterable containers of the above
                - Caikit model references (maybe?)
        streaming_parameters: The same as unary_parameters, but for streaming-input inference
            methods. All types must be in the form `Iterable[T]`

        unary_output_type (Type[DataBase]): The unary output type of the task, which all modules'
            unary-output inference methods must return. This must be a caikit data model type.
        streaming_output_type (Type[Iterable[Type[DataBase]]]): The streaming output type of the
            task, which all modules' streaming-output inference methods must return. This must be
            in the form Iterable[T].

    Returns:
        A decorator function for the task class, registering it with caikit's core registry of
            tasks.
    """

    def decorator(cls: Type[TaskBase]) -> Type[TaskBase]:
        error.subclass_check("<COR19436440E>", cls, TaskBase)

        # NB: python <= 3.9 safe way of setting class annotations
        cls_annotations = cls.__dict__.get("__annotations__", None)
        if cls_annotations is None:
            cls.__annotations__ = {}
            cls_annotations = cls.__dict__.get("__annotations__", None)

        if unary_parameters:
            cls_annotations[_UNARY_PARAMS_ANNOTATION] = unary_parameters
        if streaming_parameters:
            cls_annotations[_STREAM_PARAMS_ANNOTATION] = streaming_parameters
        if unary_output_type:
            cls_annotations[_UNARY_OUT_ANNOTATION] = unary_output_type
        if streaming_output_type:
            cls_annotations[_STREAM_OUT_ANNOTATION] = streaming_output_type

        # Backwards compatibility with old-style @tasks
        if "required_parameters" in kwargs and not unary_parameters:
            cls_annotations[_UNARY_PARAMS_ANNOTATION] = kwargs["required_parameters"]
        if "output_type" in kwargs and not unary_output_type:
            output_type = kwargs["output_type"]
            if cls._is_iterable_type(output_type):
                cls_annotations[_STREAM_OUT_ANNOTATION] = kwargs["output_type"]
            else:
                cls_annotations[_UNARY_OUT_ANNOTATION] = kwargs["output_type"]
        # End Backwards compatibility

        error.value_check(
            "<COR12671910E>",
            _UNARY_PARAMS_ANNOTATION in cls_annotations
            or _STREAM_PARAMS_ANNOTATION in cls_annotations,
            "At least one input type must be set on a task",
        )
        error.value_check(
            "<COR12671910E>",
            _UNARY_OUT_ANNOTATION in cls_annotations
            or _STREAM_OUT_ANNOTATION in cls_annotations,
            "At least one output type must be set on a task",
        )

        if _UNARY_OUT_ANNOTATION in cls_annotations:
            error.subclass_check(
                "<COR12766440E>", cls.get_output_type(output_streaming=False), DataBase
            )

        if _STREAM_OUT_ANNOTATION in cls_annotations:
            if typing.get_origin(cls.get_output_type(output_streaming=True)) is None:
                raise TypeError(
                    f"subclass check failed: {cls.get_output_type(output_streaming=True)} is \
                        not a subclass of (<class 'collections.abc.Iterable'>"
                )
            error.subclass_check(
                "<COR12766440E>",
                typing.get_origin(cls.get_output_type(output_streaming=True)),
                collections.abc.Iterable,
            )

        if _UNARY_PARAMS_ANNOTATION in cls_annotations:
            params_dict = cls.get_required_parameters(input_streaming=False)
            error.type_check("<COR19906440E>", dict, params_dict=params_dict)
            error.type_check_all(
                "<COR00123440E>", str, params_dict_keys=params_dict.keys()
            )
            # TODO: check proto-ability of things
        if _STREAM_PARAMS_ANNOTATION in cls_annotations:
            params_dict = cls.get_required_parameters(input_streaming=True)
            error.type_check("<COR19556230E>", dict, params_dict=params_dict)
            error.type_check_all(
                "<COR58796465E>", str, params_dict_keys=params_dict.keys()
            )
            for v in params_dict.values():
                error.subclass_check(
                    "<COR52740295E>", typing.get_origin(v), collections.abc.Iterable
                )

        return cls

    return decorator


def _make_keyname_for_module(module_class: Type) -> str:
    return ".".join([module_class.__module__, module_class.__qualname__])
