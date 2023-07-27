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
"""This module is responsible for creating service objects for the runtime to consume"""
# Standard
from enum import Enum
from types import ModuleType
from typing import Callable, Dict, Set, Type
import dataclasses

# Third Party
import google.protobuf.descriptor
import google.protobuf.service
import grpc

# First Party
from py_to_proto.json_to_service import json_to_service
import aconfig
import alog

# Local
from caikit import get_config
from caikit.core import ModuleBase
from caikit.interfaces.runtime.data_model import (
    TrainingInfoRequest,
    TrainingStatusResponse,
)
from caikit.runtime import service_generation
from caikit.runtime.service_generation.rpcs import CaikitRPCBase, snake_to_upper_camel
from caikit.runtime.utils import import_util
import caikit.core

log = alog.use_channel("SVC-FACTORY")

TRAINING_MANAGEMENT_SERVICE_NAME = "TrainingManagement"
TRAINING_MANAGEMENT_SERVICE_SPEC = {
    "service": {
        "rpcs": [
            {
                "name": "GetTrainingStatus",
                "input_type": TrainingInfoRequest.get_proto_class().DESCRIPTOR.full_name,
                "output_type": TrainingStatusResponse.get_proto_class().DESCRIPTOR.full_name,
            }
        ]
    }
}


@dataclasses.dataclass
class ServicePackage:
    """Container with references to:
    - A service Class
    - A service Descriptor
    - A grpc servicer registration function
    - A client stub
    - A client messages module
    """

    service: Type[google.protobuf.service.Service]
    descriptor: google.protobuf.descriptor.ServiceDescriptor
    registration_function: Callable[
        [google.protobuf.service.Service, grpc.Server], None
    ]
    stub_class: Type
    messages: ModuleType
    caikit_rpcs: Dict[str, CaikitRPCBase]


class ServicePackageFactory:
    """Factory responsible for yielding the correct concrete ServicePackage implementation"""

    class ServiceType(Enum):
        INFERENCE = 1  # Inference service for the GlobalPredictServicer
        TRAINING = 2  # Training service for the GlobalTrainServicer
        TRAINING_MANAGEMENT = 3

    @classmethod
    def get_service_package(cls, service_type: ServiceType) -> ServicePackage:
        """Public factory API. Returns a service package of the requested type, from the
        configured source.

        Args:
            service_type (ServicePackageFactory.ServiceType): The type of service to build,
                to match the servicer implementation that will handle it. e.g. the
                GlobalPredictServicer expects an "INFERENCE" service

        Returns:
            ServicePackage: A container with properties referencing everything you need to bind a
                concrete Servicer implementation to a protobufs Service and grpc Server
        """
        if service_type == cls.ServiceType.TRAINING_MANAGEMENT:
            grpc_service = json_to_service(
                name=TRAINING_MANAGEMENT_SERVICE_NAME,
                package="caikit.runtime.training",
                json_service_def=TRAINING_MANAGEMENT_SERVICE_SPEC,
            )

            return ServicePackage(
                service=grpc_service.service_class,
                descriptor=grpc_service.descriptor,
                registration_function=grpc_service.registration_function,
                stub_class=grpc_service.client_stub_class,
                messages=None,  # we don't need messages here
                caikit_rpcs={},  # No caikit RPCs
            )

        # First make sure we import the data model for the correct library
        # !!!! This will use the `caikit_library` config
        _ = import_util.get_data_model()

        caikit_config = get_config()
        lib = caikit_config.runtime.library
        ai_domain_name = snake_to_upper_camel(lib.replace("caikit_", ""))
        package_name = f"caikit.runtime.{ai_domain_name}"

        # Then do API introspection to come up with all the API definitions to support
        clean_modules = ServicePackageFactory._get_and_filter_modules(
            caikit_config, lib
        )

        if service_type == cls.ServiceType.INFERENCE:
            rpc_list = service_generation.create_inference_rpcs(clean_modules)
            service_name = f"{ai_domain_name}Service"
        else:  # service_type == cls.ServiceType.TRAINING
            rpc_list = service_generation.create_training_rpcs(clean_modules)
            service_name = f"{ai_domain_name}TrainingService"

        rpc_list = [rpc for rpc in rpc_list if rpc.return_type is not None]

        request_data_models = [
            rpc.create_request_data_model(package_name) for rpc in rpc_list
        ]

        client_module = ModuleType(
            "ClientMessages",
            "Package with service message class implementations",
        )

        for dm_class in request_data_models:
            # We need the message class that data model serializes to
            setattr(client_module, dm_class.__name__, type(dm_class().to_proto()))

        rpc_jsons = [rpc.create_rpc_json(package_name) for rpc in rpc_list]
        service_json = {"service": {"rpcs": rpc_jsons}}
        grpc_service = json_to_service(
            name=service_name, package=package_name, json_service_def=service_json
        )

        return ServicePackage(
            service=grpc_service.service_class,
            descriptor=grpc_service.descriptor,
            registration_function=grpc_service.registration_function,
            stub_class=grpc_service.client_stub_class,
            messages=client_module,
            caikit_rpcs={rpc.name: rpc for rpc in rpc_list},
        )

    # Implementation details for pure python service packages #
    @staticmethod
    def _get_and_filter_modules(
        caikit_config: aconfig.Config, lib: str
    ) -> Set[Type[ModuleBase]]:
        clean_modules = set()
        modules = [
            module_class
            for module_class in caikit.core.registries.module_registry().values()
            if module_class.__module__.partition(".")[0] == lib
        ]
        log.debug("Found all modules %s for library %s.", modules, lib)

        # Check config for any explicit inclusions
        included_task_types = (
            caikit_config.runtime.service_generation
            and caikit_config.runtime.service_generation.task_types
            and caikit_config.runtime.service_generation.task_types.included
        )
        included_modules = (
            caikit_config.runtime.service_generation
            and caikit_config.runtime.service_generation.module_guids
            and caikit_config.runtime.service_generation.module_guids.included
        )

        # Check config for any exclusions
        excluded_task_types = (
            caikit_config.runtime.service_generation
            and caikit_config.runtime.service_generation.task_types
            and caikit_config.runtime.service_generation.task_types.excluded
        )
        excluded_modules = (
            caikit_config.runtime.service_generation
            and caikit_config.runtime.service_generation.module_guids
            and caikit_config.runtime.service_generation.module_guids.excluded
        )

        for ck_module in modules:
            # Only create for modules from defined included and exclusion list

            if not ck_module.TASK_CLASS:
                log.debug(
                    "Skipping module %s with no task",
                    ck_module,
                )
                continue

            if (
                excluded_task_types
                and ck_module.TASK_CLASS.__name__ in excluded_task_types
            ):
                log.debug(
                    "Skipping module %s with excluded task %s",
                    ck_module,
                    ck_module.TASK_CLASS.__name__,
                )
                continue

            if excluded_modules and ck_module.MODULE_ID in excluded_modules:
                log.debug(
                    "Skipping module %s with excluded id %s",
                    ck_module,
                    ck_module.MODULE_ID,
                )
                continue

            # no inclusions specified means include everything
            if (included_task_types is None or included_task_types == []) and (
                included_modules is None or included_modules == []
            ):
                clean_modules.add(ck_module)

            # if inclusion is specified, use that
            else:
                if (included_modules and ck_module.MODULE_ID in included_modules) or (
                    included_task_types
                    and ck_module.TASK_CLASS.__name__ in included_task_types
                ):
                    clean_modules.add(ck_module)

        log.debug(
            "Filtered list of modules %s after excluding task types: %s and modules ids: %s. \
                Exclusions are defined in config",
            clean_modules,
            excluded_task_types,
            excluded_modules,
        )
        return clean_modules
