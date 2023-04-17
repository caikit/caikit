"""
CAIKit is an AI toolkit that enables AI users to consume stable task-specific
model APIs while enabling AI developers build algorithms and models in a
modular/composable framework.
"""

# Local
from . import core, interfaces

# Expose model management at the top level
from .core import extract, load, resolve_and_load

config = core.config.ConfigParser.get_config()
