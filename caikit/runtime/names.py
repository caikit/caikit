# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
! NOTE ! This file should not import any extra dependencies. It is intended for
use by client libraries that do not necessarily use a specific runtime server
type.
"""

# Standard
from enum import Enum
from typing import Optional, Type, Union
import re

# Third Party
from grpc import StatusCode

# First Party
import alog

# Local
from caikit.config import get_config
from caikit.core.exceptions.caikit_core_exception import CaikitCoreStatusCode
from caikit.core.modules import ModuleBase
from caikit.core.task import TaskBase
from caikit.core.toolkit.name_tools import camel_to_snake_case, snake_to_upper_camel
from caikit.interfaces.runtime.data_model import (
    DeployModelRequest,
    ModelInfo,
    ModelInfoRequest,
    ModelInfoResponse,
    RuntimeInfoRequest,
    RuntimeInfoResponse,
    TrainingInfoRequest,
    TrainingStatusResponse,
    UndeployModelRequest,
)

log = alog.use_channel("RNTM-NAMES")


################################# Service Names ################################


class ServiceType(Enum):
    """Common class for describing service types"""

    INFERENCE = 1  # Inference service for the GlobalPredictServicer
    TRAINING = 2  # Training service for the GlobalTrainServicer
    TRAINING_MANAGEMENT = 3
    INFO = 4
    MODEL_MANAGEMENT = 5


############################ Service Name Generation ###########################


##  Service Package Descriptors


def get_ai_domain() -> str:
    """Get the string name for the AI domain

    Returns:
        domain(str): The domain for this service
    """
    caikit_config = get_config()
    lib = caikit_config.runtime.library
    default_ai_domain_name = snake_to_upper_camel(lib.replace("caikit_", ""))
    ai_domain_name = (
        caikit_config.runtime.service_generation.domain or default_ai_domain_name
    )
    return ai_domain_name


def get_service_package_name(service_type: Optional[ServiceType] = None) -> str:
    """This helper will get the name of service package

    Args:
        service_type Optional[ServiceType]: The Service Type's package name to fetch defaults
            to runtime

    Returns:
        str: The name of the service package
    """

    # If specific service_type was provided then return their packages
    if service_type == ServiceType.INFO:
        return INFO_SERVICE_PACKAGE
    elif service_type == ServiceType.TRAINING_MANAGEMENT:
        return TRAINING_MANAGEMENT_SERVICE_PACKAGE
    elif service_type == ServiceType.MODEL_MANAGEMENT:
        return MODEL_MANAGEMENT_SERVICE_PACKAGE

    caikit_config = get_config()
    ai_domain_name = get_ai_domain()
    default_package_name = f"caikit.runtime.{ai_domain_name}"
    package_name = (
        caikit_config.runtime.service_generation.package or default_package_name
    )
    return package_name


def get_service_name(service_type: ServiceType) -> str:
    """This helper will get the name of the service

    Args:
        service_type ServiceType: The Service Type whose name to fetch

    Returns:
        str: The name of the service
    """
    if service_type == ServiceType.INFERENCE:
        return f"{get_ai_domain()}Service"
    elif service_type == ServiceType.TRAINING:
        return f"{get_ai_domain()}TrainingService"
    elif service_type == ServiceType.TRAINING_MANAGEMENT:
        return TRAINING_MANAGEMENT_SERVICE_NAME
    elif service_type == ServiceType.INFO:
        return INFO_SERVICE_NAME


##  Service RPC Descriptors


def get_train_rpc_name(module_class: Type[ModuleBase]) -> str:
    """Helper function to convert from the name of a module to the name of the
    request RPC function
    """

    # 🌶️🌶️🌶️ The naming scheme for training RPCs probably needs to change.
    # This uses the first task from the `tasks` kwarg in the `@caikit.module` decorator.
    # This is both:
    # - Flaky, since re-ordering that list would be perfectly reasonable and valid to do except
    #   for the side effect of breaking the training service api
    # - Not very intuitive, since a module supporting multiple tasks will have a training
    #   endpoint that lists only one of them
    rpc_name = snake_to_upper_camel(
        f"{next(iter(module_class.tasks)).__name__}_{module_class.__name__}_Train"
    )

    if len(module_class.tasks) > 1:
        log.warning(
            "<RUN35134050W>",
            "Multiple tasks detected for training rpc. "
            "Module: [%s], Tasks: [%s], RPC name: %s ",
            module_class,
            module_class.tasks,
            rpc_name,
        )

    return rpc_name


def get_task_predict_rpc_name(
    task_or_module_class: Type[Union[ModuleBase, TaskBase]],
    input_streaming: bool = False,
    output_streaming: bool = False,
) -> str:
    """Helper function to get the name of a task's RPC"""
    task_class = (
        next(iter(task_or_module_class.tasks))
        if issubclass(task_or_module_class, ModuleBase)
        else task_or_module_class
    )

    if input_streaming and output_streaming:
        return snake_to_upper_camel(f"BidiStreaming{task_class.__name__}_Predict")
    if output_streaming:
        return snake_to_upper_camel(f"ServerStreaming{task_class.__name__}_Predict")
    if input_streaming:
        return snake_to_upper_camel(f"ClientStreaming{task_class.__name__}_Predict")
    return snake_to_upper_camel(f"{task_class.__name__}_Predict")


##  Service DataModel Name Descriptors


def get_train_request_name(module_class: Type[ModuleBase]) -> str:
    """Helper function to get the request name of a Train Service"""
    return f"{get_train_rpc_name(module_class)}Request"


def get_train_parameter_name(module_class: Type[ModuleBase]) -> str:
    """Helper function to get the inner request parameter  name of a Train Service"""
    return f"{get_train_rpc_name(module_class)}Parameters"


def get_task_predict_request_name(
    task_or_module_class: Type[Union[ModuleBase, TaskBase]],
    input_streaming: bool = False,
    output_streaming: bool = False,
) -> str:
    """Helper function to get the name of an RPC's request data type"""

    task_class = (
        next(iter(task_or_module_class.tasks))
        if issubclass(task_or_module_class, ModuleBase)
        else task_or_module_class
    )

    if input_streaming and output_streaming:
        return snake_to_upper_camel(f"BidiStreaming{task_class.__name__}_Request")
    if output_streaming:
        return snake_to_upper_camel(f"ServerStreaming{task_class.__name__}_Request")
    if input_streaming:
        return snake_to_upper_camel(f"ClientStreaming{task_class.__name__}_Request")
    return snake_to_upper_camel(f"{task_class.__name__}_Request")


##  Service Definitions

TRAINING_MANAGEMENT_SERVICE_NAME = "TrainingManagement"
TRAINING_MANAGEMENT_SERVICE_PACKAGE = "caikit.runtime.training"
TRAINING_MANAGEMENT_SERVICE_SPEC = {
    "service": {
        "rpcs": [
            {
                "name": "GetTrainingStatus",
                "input_type": TrainingInfoRequest.get_proto_class().DESCRIPTOR.full_name,
                "output_type": TrainingStatusResponse.get_proto_class().DESCRIPTOR.full_name,
            },
            {
                "name": "CancelTraining",
                "input_type": TrainingInfoRequest.get_proto_class().DESCRIPTOR.full_name,
                "output_type": TrainingStatusResponse.get_proto_class().DESCRIPTOR.full_name,
            },
        ]
    }
}

INFO_SERVICE_NAME = "InfoService"
INFO_SERVICE_PACKAGE = "caikit.runtime.info"
INFO_SERVICE_SPEC = {
    "service": {
        "rpcs": [
            {
                "name": "GetRuntimeInfo",
                "input_type": RuntimeInfoRequest.get_proto_class().DESCRIPTOR.full_name,
                "output_type": RuntimeInfoResponse.get_proto_class().DESCRIPTOR.full_name,
            },
            {
                "name": "GetModelsInfo",
                "input_type": ModelInfoRequest.get_proto_class().DESCRIPTOR.full_name,
                "output_type": ModelInfoResponse.get_proto_class().DESCRIPTOR.full_name,
            },
        ]
    }
}

MODEL_MANAGEMENT_SERVICE_NAME = "ModelManagement"
MODEL_MANAGEMENT_SERVICE_PACKAGE = "caikit.runtime.models"
MODEL_MANAGEMENT_SERVICE_SPEC = {
    "service": {
        "rpcs": [
            {
                "name": "DeployModel",
                "input_type": DeployModelRequest.get_proto_class().DESCRIPTOR.full_name,
                "output_type": ModelInfo.get_proto_class().DESCRIPTOR.full_name,
            },
            {
                "name": "UndeployModel",
                "input_type": UndeployModelRequest.get_proto_class().DESCRIPTOR.full_name,
                "output_type": UndeployModelRequest.get_proto_class().DESCRIPTOR.full_name,
            },
        ]
    }
}

################################# Server Names #################################

# Invocation metadata key for the model ID, provided by Model Mesh
MODEL_MESH_MODEL_ID_KEY = "mm-model-id"


## HTTP Server

# Endpoint to use for health checks
HEALTH_ENDPOINT = "/health"

# Endpoint to use for server info
INFO_ENDPOINT = "/info"
RUNTIME_INFO_ENDPOINT = f"{INFO_ENDPOINT}/version"
MODELS_INFO_ENDPOINT = f"{INFO_ENDPOINT}/models"

# Endpoints to use for resource management
MANAGEMENT_ENDPOINT = "/management"
MODEL_MANAGEMENT_ENDPOINT = f"{MANAGEMENT_ENDPOINT}/models"
TRAINING_MANAGEMENT_ENDPOINT = f"{MANAGEMENT_ENDPOINT}/trainings"

# These keys are used to define the logical sections of the request and response
# data structures.
REQUIRED_INPUTS_KEY = "inputs"
OPTIONAL_INPUTS_KEY = "parameters"
MODEL_ID = "model_id"

# Stream event type for HTTP output streaming
class StreamEventTypes(Enum):
    MESSAGE = "message"
    ERROR = "error"


def get_http_route_name(rpc_name: str) -> str:
    """Function to get the http route for a given rpc name

    Args:
        rpc_name (str): The name of the Caikit RPC

    Raises:
        NotImplementedError: If the RPC is not a Train or Predict RPC

    Returns:
        str: The name of the http route for RPC
    """
    if rpc_name.endswith("Predict"):
        task_name = camel_to_snake_case(
            re.sub("Task$", "", re.sub("Predict$", "", rpc_name)),
            kebab_case=True,
        )
        route = "/".join([get_config().runtime.http.route_prefix, "task", task_name])
        if route[0] != "/":
            route = "/" + route
        return route
    if rpc_name.endswith("Train"):
        route = "/".join([get_config().runtime.http.route_prefix, rpc_name])
        if route[0] != "/":
            route = "/" + route
        return route
    raise NotImplementedError(f"Unknown RPC type for rpc name {rpc_name}")


## GRPC Server


def get_grpc_route_name(service_type: ServiceType, rpc_name: str) -> str:
    """Function to get GRPC name for a given service type and rpc name

    Args:
        rpc_name (str): The name of the Caikit RPC

    Returns:
        str: The name of the GRPC route for RPC
    """
    return f"/{get_service_package_name(service_type)}.{get_service_name(service_type)}/{rpc_name}"


## Status Code Mappings

STATUS_CODE_TO_HTTP = {
    # Mapping from GRPC codes to their corresponding HTTP codes
    # pylint: disable=line-too-long
    # CITE: https://chromium.googlesource.com/external/github.com/grpc/grpc/+/refs/tags/v1.21.4-pre1/doc/statuscodes.md
    StatusCode.OK: 200,
    StatusCode.INVALID_ARGUMENT: 400,
    StatusCode.FAILED_PRECONDITION: 400,
    StatusCode.OUT_OF_RANGE: 400,
    StatusCode.UNAUTHENTICATED: 401,
    StatusCode.PERMISSION_DENIED: 403,
    StatusCode.NOT_FOUND: 404,
    StatusCode.ALREADY_EXISTS: 409,
    StatusCode.ABORTED: 409,
    StatusCode.RESOURCE_EXHAUSTED: 429,
    StatusCode.CANCELLED: 499,
    StatusCode.UNKNOWN: 500,
    StatusCode.DATA_LOSS: 500,
    StatusCode.UNIMPLEMENTED: 501,
    StatusCode.UNAVAILABLE: 501,
    StatusCode.DEADLINE_EXCEEDED: 504,
    # Mapping from CaikitCore StatusCodes codes to their corresponding HTTP codes
    CaikitCoreStatusCode.INVALID_ARGUMENT: 400,
    CaikitCoreStatusCode.UNAUTHORIZED: 401,
    CaikitCoreStatusCode.FORBIDDEN: 403,
    CaikitCoreStatusCode.NOT_FOUND: 404,
    CaikitCoreStatusCode.CONNECTION_ERROR: 500,
    CaikitCoreStatusCode.UNKNOWN: 500,
    CaikitCoreStatusCode.FATAL: 500,
}

# Invert STATUS_CODE_TO_HTTP preferring grpc.StatusCodes over CaikitCoreStatusCode
# this is because CaikitRuntimeExceptions expect StatusCode and not the caikit version
HTTP_TO_STATUS_CODE = {}
for key, val in STATUS_CODE_TO_HTTP.items():
    if val not in HTTP_TO_STATUS_CODE or isinstance(key, StatusCode):
        HTTP_TO_STATUS_CODE[val] = key
