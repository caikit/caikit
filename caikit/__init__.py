"""
CAIKit is an AI toolkit that enables AI users to consume stable task-specific
model APIs while enabling AI developers build algorithms and models in a
modular/composable framework.
"""

# First Party
import aconfig

# Local
from . import config, core, interfaces

# Expose configuration fn at the top level
from .config import configure

# Expose model management at the top level
from .core import extract, load, resolve_and_load

# Define a `config` attribute that is written to by `configure()`
config: aconfig.Config
