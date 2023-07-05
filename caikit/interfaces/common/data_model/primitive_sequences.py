# Standard
from typing import List

# Local
from caikit.core.data_model import DataObjectBase, dataobject
from caikit.interfaces.runtime.data_model.training_management import RUNTIME_PACKAGE


# interfaces required for unions of lists
@dataobject(RUNTIME_PACKAGE)
class IntSequence(DataObjectBase):
    values: List[int]


@dataobject(RUNTIME_PACKAGE)
class FloatSequence(DataObjectBase):
    values: List[float]


@dataobject(RUNTIME_PACKAGE)
class StrSequence(DataObjectBase):
    values: List[str]


@dataobject(RUNTIME_PACKAGE)
class BoolSequence(DataObjectBase):
    values: List[bool]
