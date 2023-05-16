# Standard
import os

# Local
from . import data_model, modules
from .modules import InnerModule, OtherModule, SampleModule, SamplePrimitiveModule
from caikit.config import configure

# Run configure for sample_lib configuration
configure(os.path.join(os.path.dirname(__file__), "config.yml"))
