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
from typing import Callable, Dict, List, Set, Type, Union

# First Party
from alog import alog

# Local
from caikit.core.data_model import DataStream
from caikit.core.data_model.base import DataBase
from caikit.core.toolkit.errors import error_handler

log = alog.use_channel("TASK_BASE")
error = error_handler.get(log)

ProtoableInputTypes = Type[Union[int, float, str, bytes, bool, DataBase]]
ValidInputTypes = Union[
    ProtoableInputTypes, List[ProtoableInputTypes], DataStream[ProtoableInputTypes]
]


class TaskGroupBase:
    @classmethod
    def validate_task_inputs(cls) -> bool:
        pass

    @classmethod
    def get_input_type_set(cls) -> Set[ProtoableInputTypes]:
        raise NotImplementedError("This is implemented by the @taskgroup decorator!")


class TaskBase:
    @classmethod
    def validate_run_signature(cls) -> bool:
        # TODO: implement
        pass

    @classmethod
    def get_required_inputs(cls) -> Dict[str, ValidInputTypes]:
        raise NotImplementedError("This is implemented by the @task decorator!")

    @classmethod
    def get_output_type(cls) -> Type[DataBase]:
        raise NotImplementedError("This is implemented by the @task decorator!")

    @classmethod
    def get_task_group(cls) -> Type[TaskGroupBase]:
        raise NotImplementedError("This is implemented by the @task decorator!")


def task(
    task_group: Type[TaskGroupBase],
    required_inputs: Dict[str, ValidInputTypes],
    output_type: Type[DataBase],
) -> Callable[[Type[TaskBase]], Type[TaskBase]]:
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
        task_group (Type[TaskGroupBase]): The AI Task Group that this task belongs to
        required_inputs (Dict[str, ValidInputTypes]): The required parameters that all public
            models' .run functions must contain. A dictionary of parameter name to parameter
            type, where the types can be in the set of:
                - Python primitives
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
    if not issubclass(output_type, DataBase):
        raise ValueError("output_type must be a data model")
    if not issubclass(task_group, TaskGroupBase):
        raise ValueError("task_group must be a TaskGroup class")

    for parameter_name, input_type in required_inputs.items():
        if input_type not in task_group.get_input_type_set():
            raise ValueError(
                f"Task parameter {parameter_name} has type {input_type} not in task_group: "
                f"{task_group.__name__}. Valid types are: {task_group.get_input_type_set()}"
            )

    def get_required_inputs(_):
        return required_inputs

    def get_output_type(_):
        return output_type

    def get_task_group(_):
        return task_group

    def decorator(cls: Type[TaskBase]) -> Type[TaskBase]:
        if not isinstance(cls, type) or not issubclass(cls, TaskBase):
            raise ValueError("decorated class must extend TaskBase")
        setattr(cls, "get_required_inputs", classmethod(get_required_inputs))
        setattr(cls, "get_output_type", classmethod(get_output_type))
        setattr(cls, "get_task_group", classmethod(get_task_group))
        return cls

    return decorator


def taskgroup(
    input_types: Set[ProtoableInputTypes],
) -> Callable[[Type[TaskGroupBase]], Type[TaskGroupBase]]:
    """The decorator for AI Task Groups"""

    def type_check(x: type) -> bool:
        return (
            x == int
            or x == float
            or x == str
            or x == bytes
            or x == bool
            or (isinstance(x, type) and issubclass(x, DataBase))
        )

    for input_type in input_types:
        error.value_check(
            "<COR98288712E>",
            type_check(input_type),
            input_type,
            msg="TaskGroup inputs must be python primitive types or data model types. Got {}",
        )

    def get_input_type_set(_) -> Set[ProtoableInputTypes]:
        return input_types

    def decorator(cls: Type[TaskGroupBase]) -> Type[TaskGroupBase]:
        error.value_check(
            "<COR98211745E>",
            isinstance(cls, type) and issubclass(cls, TaskGroupBase),
            cls,
            msg="@taskgroup class must extend TaskGroupBase",
        )
        setattr(cls, "get_input_type_set", classmethod(get_input_type_set))
        return cls

    return decorator
