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
from typing import Callable, Dict, List, Type, Union

# First Party
from alog import alog

# Local
from .data_model import DataStream
from .data_model.base import DataBase
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
    def validate_run_signature(cls) -> bool:
        # TODO: implement
        pass

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


def task(
    required_parameters: Dict[str, ValidInputTypes],
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
        required_parameters (Dict[str, ValidInputTypes]): The required parameters that all public
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
    error.subclass_check("<COR12766440E>", output_type, DataBase)

    def get_required_parameters(_):
        return required_parameters

    def get_output_type(_):
        return output_type

    def decorator(cls: Type[TaskBase]) -> Type[TaskBase]:
        get_required_parameters.__doc__ = f"""
        Returns the set of input parameters required in the `run` function for any module that
        implements the '{cls.__name__}' Task.
        
        ({required_parameters})

        Returns: Dict[str, Type]
            The parameter dictionary for the {cls.__name__} inference task
        """

        get_output_type.__doc__ = f"""
        Returns the output type required of the `run` function in any module that implements the
        {cls.__name__} task. 
        
        ({output_type})
        
        Returns: Type
            The output type of the {cls.__name__} inference task
        """
        error.subclass_check("<COR19436440E>", cls, TaskBase)
        setattr(cls, "get_required_parameters", classmethod(get_required_parameters))
        setattr(cls, "get_output_type", classmethod(get_output_type))
        return cls

    return decorator
