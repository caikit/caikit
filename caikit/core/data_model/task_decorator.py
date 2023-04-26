
# Standard
from typing import Dict, Set, Type
import caikit

from caikit.core.data_model.base import DataBase

# Local


def task(required_inputs: Dict[str, Type], output_type: Type[DataBase]):
    # TODO validate_inputs_as_raw_or_dm
    # error.value_check(
    #         "<COR95184230E>",
    #         not issubclass(cls, DataBase),
    #         "{} should not directly inherit from DataBase when using @schema",
    #         cls.__name__,
    #     )
    
    def validate_run_signature():
        pass

    # def get_all_modules(cls) -> Set[Type[caikit.core.ModuleBase]]:
    #     pass

    def get_required_inputs(cls):
        return required_inputs
    def get_output_type(cls):
        return output_type

    def decorator(cls):
        setattr(cls, "validate_run_signature", classmethod(validate_run_signature))
        # setattr(cls, "get_all_modules", classmethod(get_all_modules))
        setattr(cls, "get_required_inputs", classmethod(get_required_inputs))
        setattr(cls, "get_output_type", classmethod(get_output_type))
        return cls




    return decorator
