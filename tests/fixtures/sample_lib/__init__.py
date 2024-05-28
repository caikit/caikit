# Standard
import os

# Local
from . import data_model, modules
from .modules import (
    CompositeModule,
    InnerModule,
    MultiTaskModule,
    OtherModule,
    SampleModule,
    SamplePrimitiveModule,
    StreamingModule,
)
from caikit.config import configure

# Run configure for sample_lib configuration
configure(os.path.join(os.path.dirname(__file__), "config.yml"))
