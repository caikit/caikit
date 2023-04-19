# Standard
import os

# Local
from . import blocks, data_model, workflows
from .blocks import InnerBlock, OtherBlock, SampleBlock, SamplePrimitiveBlock
from .workflows import SampleWorkflow
from caikit.config import configure

# Run configure for sample_lib configuration
configure(os.path.join(os.path.dirname(__file__), "config.yml"))
