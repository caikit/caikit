# Local
from . import create_service
from .create_service import (
    assert_compatible,
    create_inference_rpcs,
    create_training_rpcs,
)
from .proto_package import get_ai_domain, get_runtime_service_package
