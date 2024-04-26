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
from types import ModuleType
from typing import Callable, Dict, Set, Type, Union
import dataclasses
import json
import os

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
from caikit.core import LocalBackend, ModuleBase, registries
from caikit.core.data_model.base import DataBase
from caikit.core.data_model.dataobject import _AUTO_GEN_PROTO_CLASSES
from caikit.core.exceptions import error_handler
from caikit.core.task import TaskBase
from caikit.runtime import service_generation
from caikit.runtime.names import (
    INFO_SERVICE_NAME,
    INFO_SERVICE_PACKAGE,
    INFO_SERVICE_SPEC,
    MODEL_MANAGEMENT_SERVICE_NAME,
    MODEL_MANAGEMENT_SERVICE_PACKAGE,
    MODEL_MANAGEMENT_SERVICE_SPEC,
    TRAINING_MANAGEMENT_SERVICE_NAME,
    TRAINING_MANAGEMENT_SERVICE_PACKAGE,
    TRAINING_MANAGEMENT_SERVICE_SPEC,
)
from caikit.runtime.names import ServiceType as InterfaceServiceType
from caikit.runtime.names import (
    get_service_name,
    get_service_package_name,
    get_task_predict_request_name,
    get_train_parameter_name,
    get_train_request_name,
)
from caikit.runtime.service_generation.rpcs import CaikitRPCBase
from caikit.runtime.utils import import_util

log = alog.use_channel("SVC-FACTORY")
error = error_handler.get(log)


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

    ServiceType = InterfaceServiceType

    @classmethod
    def get_service_package(
        cls, service_type: ServiceType, write_modules_file: bool = False
    ) -> ServicePackage:
        """Public factory API. Returns a service package of the requested type, from the
        configured source.

        Args:
            service_type (ServicePackageFactory.ServiceType): The type of service to build,
                to match the servicer implementation that will handle it. e.g. the
                GlobalPredictServicer expects an "INFERENCE" service
            write_modules_file (bool): if set, write out a modules.json file to list the
                included modules in this service generation. See config to customize file name at
                runtime.service_generation.backwards_compatibility.current_modules_path

        Returns:
            ServicePackage: A container with properties referencing everything you need to bind a
                concrete Servicer implementation to a protobufs Service and grpc Server
        """
        if service_type == cls.ServiceType.TRAINING_MANAGEMENT:
            grpc_service = json_to_service(
                name=TRAINING_MANAGEMENT_SERVICE_NAME,
                package=TRAINING_MANAGEMENT_SERVICE_PACKAGE,
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

        if service_type == cls.ServiceType.MODEL_MANAGEMENT:
            grpc_service = json_to_service(
                name=MODEL_MANAGEMENT_SERVICE_NAME,
                package=MODEL_MANAGEMENT_SERVICE_PACKAGE,
                json_service_def=MODEL_MANAGEMENT_SERVICE_SPEC,
            )

            return ServicePackage(
                service=grpc_service.service_class,
                descriptor=grpc_service.descriptor,
                registration_function=grpc_service.registration_function,
                stub_class=grpc_service.client_stub_class,
                messages=None,  # we don't need messages here
                caikit_rpcs={},  # No caikit RPCs
            )

        if service_type == cls.ServiceType.INFO:
            grpc_service = json_to_service(
                name=INFO_SERVICE_NAME,
                package=INFO_SERVICE_PACKAGE,
                json_service_def=INFO_SERVICE_SPEC,
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

        # Get the names for the AI domain and the proto package
        package_name = get_service_package_name(service_type)

        # Then do API introspection to come up with all the API definitions to support
        caikit_config = get_config()
        clean_modules = ServicePackageFactory._get_and_filter_modules(
            caikit_config, caikit_config.runtime.library, write_modules_file
        )

        service_name = get_service_name(service_type)
        if service_type == cls.ServiceType.INFERENCE:
            # Assert for backwards compatibility, if enabled, when service type is INFERENCE
            ServicePackageFactory._check_backwards_compatibility(
                caikit_config, clean_modules
            )

            rpc_list = service_generation.create_inference_rpcs(
                clean_modules, caikit_config
            )
        else:  # service_type == cls.ServiceType.TRAINING
            rpc_list = service_generation.create_training_rpcs(clean_modules)

        rpc_list = [rpc for rpc in rpc_list if rpc.return_type is not None]

        for rpc in rpc_list:
            rpc.create_request_data_model(package_name)

        client_module = ModuleType(
            "ClientMessages",
            "Package with service message class implementations",
        )

        for proto_class in _AUTO_GEN_PROTO_CLASSES:
            # We need all the DM objects in the client_module for ease of use
            setattr(client_module, proto_class.DESCRIPTOR.name, proto_class)

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
    def _check_backwards_compatibility(
        caikit_config: aconfig.Config, clean_modules: Set[Type[ModuleBase]]
    ):
        backwards_compat_conf = (
            caikit_config.runtime.service_generation.backwards_compatibility
        )
        if backwards_compat_conf and backwards_compat_conf.enabled:
            previous_included_modules = set()
            prev_modules_path = backwards_compat_conf.prev_modules_path
            error.value_check(
                "<SVC98335764E>",
                os.path.exists(prev_modules_path) and os.path.isfile(prev_modules_path),
                "prev_modules_path {} is not a valid file path or is missing permissions",
                prev_modules_path,
            )
            with open(prev_modules_path, encoding="utf-8") as f:
                previous_modules = json.load(f)
                previous_included_task_map = previous_modules["included_modules"]
                for task_module in previous_included_task_map.values():
                    previous_included_modules.update(task_module.keys())

            service_generation.assert_compatible(
                [mod.MODULE_ID for mod in clean_modules],
                previous_included_modules,
            )

    @staticmethod
    def _get_and_filter_modules(
        caikit_config: aconfig.Config, lib: str, write_modules_file: bool
    ) -> Set[Type[ModuleBase]]:
        clean_modules = set()
        modules = [
            module_class
            for module_class in registries.module_registry().values()
            if module_class.__module__.partition(".")[0] == lib
        ]
        # NB: The `module_registry` only includes the `LOCAL` backend modules.
        # Implementations of the same module for different backends need to be fetched from the
        # backend registry
        backend_modules = []
        for backend_dict in registries.module_backend_registry().values():
            for backend, config in backend_dict.items():
                if backend != LocalBackend.backend_type:
                    backend_modules.append(config.impl_class)
        modules.extend(backend_modules)
        log.debug("Found all modules %s for library %s.", modules, lib)

        # Check config for any explicit inclusions
        included_modules = (
            caikit_config.runtime.service_generation
            and caikit_config.runtime.service_generation.module_guids
            and caikit_config.runtime.service_generation.module_guids.included
        )

        # Check config for any exclusions
        excluded_modules = (
            caikit_config.runtime.service_generation
            and caikit_config.runtime.service_generation.module_guids
            and caikit_config.runtime.service_generation.module_guids.excluded
        )

        for ck_module in modules:
            # Only create for modules from defined included and exclusion list

            if not ck_module.tasks:
                log.debug(
                    "Skipping module %s with no tasks",
                    ck_module,
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
            if included_modules is None or included_modules == []:
                clean_modules.add(ck_module)

            # if inclusion is specified, use that
            else:
                if included_modules and ck_module.MODULE_ID in included_modules:
                    clean_modules.add(ck_module)

        log.debug(
            "Filtered list of modules %s after excluding modules ids: %s. \
                Exclusions are defined in config",
            clean_modules,
            excluded_modules,
        )

        # if enabled, write out the inclusions to modules.json
        backwards_compat_conf = (
            caikit_config.runtime.service_generation.backwards_compatibility
        )
        if write_modules_file:
            modules_json_path = (
                backwards_compat_conf and backwards_compat_conf.current_modules_path
            ) or "modules.json"
            included_dict = {}
            for module in clean_modules:
                for task_type in module.tasks:
                    included_dict.setdefault(task_type.__name__, {})[
                        module.MODULE_ID
                    ] = str(module)
            modules_dict = {
                "included_modules": included_dict,
            }
            with open(modules_json_path, "w", encoding="utf-8") as f:
                json.dump(modules_dict, f, indent=4, sort_keys=True)
        return clean_modules


def get_inference_request(
    task_or_module_class: Type[Union[ModuleBase, TaskBase]],
    input_streaming: bool = False,
    output_streaming: bool = False,
) -> Type[DataBase]:
    """Helper function to return the inference request DataModel for the Module or Task Class"""
    error.subclass_check(
        "<SVC98285724E>",
        task_or_module_class,
        ModuleBase,
        TaskBase,
    )

    request_class_name = get_task_predict_request_name(
        task_or_module_class,
        input_streaming=input_streaming,
        output_streaming=output_streaming,
    )
    log.debug(
        "Request class name %s for class %s.", request_class_name, task_or_module_class
    )
    return DataBase.get_class_for_name(request_class_name)


def get_train_request(module_class: Type[ModuleBase]) -> Type[DataBase]:
    """Helper function to return the train request DataModel for the Module Class"""
    error.subclass_check(
        "<SVC32285724E>",
        module_class,
        ModuleBase,
    )
    request_class_name = get_train_request_name(module_class)
    log.debug("Request class name %s for module %s.", request_class_name, module_class)
    return DataBase.get_class_for_name(request_class_name)


def get_train_params(module_class: Type[ModuleBase]) -> Type[DataBase]:
    """Helper function to return the train parameters DataModel for the Module Class"""
    error.subclass_check(
        "<SVC98435724E>",
        module_class,
        ModuleBase,
    )

    request_class_name = get_train_parameter_name(module_class)
    log.debug("Request class name %s for module %s.", request_class_name, module_class)
    return DataBase.get_class_for_name(request_class_name)
