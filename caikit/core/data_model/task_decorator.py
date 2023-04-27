# Standard
from typing import Dict, List, Set, Type, Union
import abc

# First Party
from alog import alog

# Local
from ..toolkit.errors import error_handler
from caikit.core.data_model import DataStream
from caikit.core.data_model.base import DataBase

log = alog.use_channel("TASK_BASE")
error = error_handler.get(log)

ProtoableInputTypes = Union[
    Type[int], Type[float], Type[str], Type[bytes], Type[bool], Type[DataBase]
]
ValidInputTypes = Union[
    ProtoableInputTypes, List[ProtoableInputTypes], DataStream[ProtoableInputTypes]
]


class TaskBase(abc.ABC):
    @classmethod
    def validate_run_signature(cls) -> bool:
        # TODO: implement
        pass

    @classmethod
    @abc.abstractmethod
    def get_required_inputs(cls) -> Dict[str, ValidInputTypes]:
        pass

    @classmethod
    @abc.abstractmethod
    def get_output_type(cls) -> Type[DataBase]:
        pass


def task(required_inputs: Dict[str, ValidInputTypes], output_type: Type[DataBase]):
    """The decorator for AI Task classes.

    This defines an output data model type for the task, and a minimal set of required inputs that all public models
    implementing this task must accept.

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
                device: caikit.interfaces.common.HardwareEnum) -> caikit.interfaces.nlp.SentimentPrediction:
            # impl

    Note the run function may include other arguments beyond the minimal required inputs for the task.

    Args:
        required_inputs (Dict[str, ValidInputTypes]): The required parameters that all public models' .run functions
            must contain. A dictionary of parameter name to parameter type, where the types can be in the set of:
                - Python primitives
                - Caikit data models
                - Iterable containers of the above
                - Caikit model references (maybe?)
        output_type (Type[DataBase]): The output type of the task, which all public models' .run functions must return.
            This must be a caikit data model type.

    Returns:
        A decorator function for the task class, registering it with caikit's core registry of tasks.
    """
    # whoops, type checking won't handle this for us
    # error.type_check("<COR98211745E>", Dict[str, ValidInputTypes], required_inputs=required_inputs)
    error.type_check("<COR98211745E>", type(DataBase), output_type=output_type)

    # def get_all_modules(cls) -> Set[Type[caikit.core.ModuleBase]]:
    #     pass

    def get_required_inputs(cls):
        return required_inputs

    def get_output_type(cls):
        return output_type

    def decorator(cls: Type[TaskBase]):
        error.type_check("<COR98211745E>", type(TaskBase), cls=cls)
        setattr(cls, "get_required_inputs", classmethod(get_required_inputs))
        setattr(cls, "get_output_type", classmethod(get_output_type))
        return cls

    return decorator
