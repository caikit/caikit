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
This file contains interfaces and descriptor functions for the runtime servers
"""

# Standard
import re
from enum import Enum

# Local
from caikit.config import get_config
from caikit.interfaces.runtime.service import (
    ServiceType,
    get_service_name,
    get_service_package_name,
)

### HTTP Server

# Endpoint to use for health checks
HEALTH_ENDPOINT = "/health"

# Endpoint to use for server info
RUNTIME_INFO_ENDPOINT = "/info/version"
MODELS_INFO_ENDPOINT = "/info/models"

# These keys are used to define the logical sections of the request and response
# data structures.
REQUIRED_INPUTS_KEY = "inputs"
OPTIONAL_INPUTS_KEY = "parameters"
MODEL_ID = "model_id"

# Stream event types enum
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
        task_name = re.sub(
            r"(?<!^)(?=[A-Z])",
            "-",
            re.sub("Task$", "", re.sub("Predict$", "", rpc_name)),
        ).lower()
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


### GRPC Server


def get_grpc_route_name(service_type: ServiceType, rpc_name: str) -> str:
    """Function to get GRPC name for a given service type and rpc name

    Args:
        rpc_name (str): The name of the Caikit RPC 

    Returns:
        str: The name of the GRPC route for RPC
    """
    return f"/{get_service_package_name(service_type)}.{get_service_name(service_type)}/{rpc_name}"
