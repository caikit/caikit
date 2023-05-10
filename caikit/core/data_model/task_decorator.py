# Standard
from typing import Dict, List, Set, Type, Union
import abc

# First Party
from alog import alog

# Local
from ..toolkit.errors import error_handler
from caikit.core.data_model import DataStream
from caikit.core.data_model.base import DataBase
import caikit

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
    # whoops, type checking won't handle this for us
    # error.type_check("<COR98211745E>", Dict[str, ValidInputTypes], required_inputs=required_inputs)
    # error.type_check("<COR98211745E>", Type[DataBase], output_type=output_type)

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
