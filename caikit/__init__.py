"""
CAIKit is an AI toolkit that enables AI users to consume stable task-specific
model APIs while enabling AI developers build algorithms and models in a
modular/composable framework.
"""
# Local
from . import core, interfaces

# Expose configuration fn and getter at the top level
from .config import configure, get_config

# Expose model management at the top level
from .core import extract, get_model_future, load, module, resolve_and_load, train
