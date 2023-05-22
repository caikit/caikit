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
from typing import Callable, Set, Type
import dataclasses
import inspect

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
    TrainingInfoResponse,
)
from caikit.runtime import service_generation
from caikit.runtime.service_generation.core_module_helpers import get_module_info
from caikit.runtime.service_generation.rpcs import snake_to_upper_camel
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
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
                "output_type": TrainingInfoResponse.get_proto_class().DESCRIPTOR.full_name,
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


class ServicePackageFactory:
    """Factory responsible for yielding the correct concrete ServicePackage implementation"""

    class ServiceType(Enum):
        INFERENCE = 1  # Inference service for the GlobalPredictServicer
        TRAINING = 2  # Training service for the GlobalTrainServicer
        TRAINING_MANAGEMENT = 3

    class ServiceSource(Enum):
        COMPILED = 1  # Pull from a protoc-compiled _pb2 module
        GENERATED = 2  # Generate a service definition by inspecting the library's APIs

    @classmethod
    def get_service_package(
        cls,
        service_type: ServiceType,
        source: ServiceSource,
    ) -> ServicePackage:
        """Public factory API. Returns a service package of the requested type, from the
        configured source.

        Args:
            service_type (ServicePackageFactory.ServiceType): The type of service to build,
                to match the servicer implementation that will handle it. e.g. the
                GlobalPredictServicer expects an "INFERENCE" service
            source (ServicePackageFactory.ServiceSource): Describes where the service artifacts
                should be pulled from or how they should be constructed

        Returns:
            ServicePackage: A container with properties referencing everything you need to bind a
                concrete Servicer implementation to a protobufs Service and grpc Server
        """
        if source == cls.ServiceSource.COMPILED:
            # Use our import_utils to extract the correct bits out of a set of compiled pb2
            # packages
            lib_name = cls._get_lib_name_for_servicer()

            if service_type == cls.ServiceType.INFERENCE:
                # 🌶️🌶️🌶️! hardcoded service names
                compiled_pb2_package = cls._get_compiled_proto_module(
                    "caikit_runtime_pb2"
                )
                compiled_pb2_grpc_package = cls._get_compiled_proto_module(
                    "caikit_runtime_pb2_grpc"
                )
            elif service_type == cls.ServiceType.TRAINING_MANAGEMENT:
                raise CaikitRuntimeException(
                    grpc.StatusCode.INTERNAL,
                    "Not allowed to get Training Management services from compiled packages",
                )
            else:  # elif  service_type == cls.ServiceType.TRAINING:
                # (using final _else_ for static analysis happiness)
                # 🌶️🌶️🌶️! hardcoded service names
                compiled_pb2_package = cls._get_compiled_proto_module(
                    "caikit_runtime_train_pb2"
                )
                compiled_pb2_grpc_package = cls._get_compiled_proto_module(
                    "caikit_runtime_train_pb2_grpc"
                )

            # Dynamically create a new module to hold all the service's messages
            client_module = ModuleType(
                "ClientMessages", "Package with service message class implementations"
            )
            for k, v in compiled_pb2_package.__dict__.items():
                if inspect.isclass(v) and issubclass(
                    v, google.protobuf.message.Message
                ):
                    setattr(client_module, k, v)

            return ServicePackage(
                service=cls._get_servicer_class(compiled_pb2_grpc_package, lib_name),
                descriptor=cls._get_service_descriptor(compiled_pb2_package, lib_name),
                registration_function=cls._get_servicer_function(
                    compiled_pb2_grpc_package, lib_name
                ),
                stub_class=cls._get_servicer_stub(compiled_pb2_grpc_package, lib_name),
                messages=client_module,
            )

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
            )

        if source == cls.ServiceSource.GENERATED:
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
                task_rpc_list = service_generation.create_inference_rpcs(clean_modules)
                service_name = f"{ai_domain_name}Service"
            else:  # service_type == cls.ServiceType.TRAINING
                task_rpc_list = service_generation.create_training_rpcs(clean_modules)
                service_name = f"{ai_domain_name}TrainingService"

            task_rpc_list = [
                rpc for rpc in task_rpc_list if rpc.return_type is not None
            ]

            request_data_models = [
                rpc.create_request_data_model(package_name) for rpc in task_rpc_list
            ]

            client_module = ModuleType(
                "ClientMessages",
                "Package with service message class implementations",
            )

            for dm_class in request_data_models:
                # We need the message class that data model serializes to
                setattr(client_module, dm_class.__name__, type(dm_class().to_proto()))

            rpc_jsons = [rpc.create_rpc_json(package_name) for rpc in task_rpc_list]
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
            module_info = get_module_info(ck_module)
            if excluded_task_types and module_info.type in excluded_task_types:
                log.debug("Skipping module %s of type %s", ck_module, module_info.type)
                continue

            if excluded_modules and ck_module.MODULE_ID in excluded_modules:
                log.debug("Skipping module %s of id %s", ck_module, ck_module.MODULE_ID)
                continue

            # no inclusions specified means include everything
            if (included_task_types is None or included_task_types == []) and (
                included_modules is None or included_modules == []
            ):
                clean_modules.add(ck_module)

            # if inclusion is specified, use that
            else:
                if (included_modules and ck_module.MODULE_ID in included_modules) or (
                    included_task_types and module_info.type in included_task_types
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

    # Implementation Details for protoc-compiled packages #
    @staticmethod
    def _get_service_descriptor(
        caikit_runtime_pb2,
        lib_name,
    ) -> google.protobuf.descriptor.ServiceDescriptor:
        """Get Service descriptor from caikit_runtime_pb2 module"""
        service = f"_{lib_name.upper()}SERVICE"
        train_service = f"_{lib_name.upper()}TRAININGSERVICE"

        if hasattr(caikit_runtime_pb2, service):
            return getattr(caikit_runtime_pb2, service)
        if hasattr(caikit_runtime_pb2, train_service):
            return getattr(caikit_runtime_pb2, train_service)

        raise CaikitRuntimeException(
            grpc.StatusCode.INTERNAL,
            "Could not find service descriptor in caikit_runtime_pb2",
        )

    @staticmethod
    def _get_servicer_function(
        caikit_runtime_pb2_grpc,
        lib_name,
    ) -> Callable[[google.protobuf.service.Service, grpc.Server], None]:
        """Get ServiceServicer function from caikit_runtime_pb2_grpc module"""
        servicer = f"add_{lib_name}ServiceServicer_to_server"
        train_servicer = f"add_{lib_name}TrainingServiceServicer_to_server"

        if hasattr(caikit_runtime_pb2_grpc, servicer):
            return getattr(caikit_runtime_pb2_grpc, servicer)
        if hasattr(caikit_runtime_pb2_grpc, train_servicer):
            return getattr(caikit_runtime_pb2_grpc, train_servicer)

        raise CaikitRuntimeException(
            grpc.StatusCode.INTERNAL,
            "Could not find servicer function in caikit_runtime_pb2_grpc",
        )

    @staticmethod
    def _get_servicer_class(
        caikit_runtime_pb2_grpc,
        lib_name,
    ) -> Type[google.protobuf.service.Service]:
        """Get google.protobufs.service.Service interface class from
        caikit_runtime_pb2_grpc module"""
        servicer = f"{lib_name}ServiceServicer"
        train_servicer = f"{lib_name}TrainingServiceServicer"

        if hasattr(caikit_runtime_pb2_grpc, servicer):
            return getattr(caikit_runtime_pb2_grpc, servicer)
        if hasattr(caikit_runtime_pb2_grpc, train_servicer):
            return getattr(caikit_runtime_pb2_grpc, train_servicer)

        raise CaikitRuntimeException(
            grpc.StatusCode.INTERNAL,
            f"Could not find servicer class {servicer} or {train_servicer} "
            "in caikit_runtime_pb2_grpc",
        )

    @staticmethod
    def _get_servicer_stub(
        caikit_runtime_pb2_grpc,
        lib_name,
    ) -> type:
        """Get ServiceStub class from caikit_runtime_pb2_grpc module"""
        servicer = f"{lib_name}ServiceStub"
        train_servicer = f"{lib_name}TrainingServiceStub"

        if hasattr(caikit_runtime_pb2_grpc, servicer):
            return getattr(caikit_runtime_pb2_grpc, servicer)
        if hasattr(caikit_runtime_pb2_grpc, train_servicer):
            return getattr(caikit_runtime_pb2_grpc, train_servicer)

        raise CaikitRuntimeException(
            grpc.StatusCode.INTERNAL,
            "Could not find servicer stub in caikit_runtime_pb2_grpc",
        )

    @staticmethod
    def _get_lib_name_for_servicer() -> str:
        """Get caikit library name from Config, make upper case and not include caikit_"""
        lib_names = import_util.clean_lib_names(get_config().runtime.library)
        assert len(lib_names) == 1, "Only 1 caikit library supported for now"
        return snake_to_upper_camel(lib_names[0].replace("caikit_", ""))

    @staticmethod
    def _get_compiled_proto_module(
        module: str,
        config=None,
    ) -> ModuleType:
        """
        Dynamically import the compiled service module. This is accomplished via dynamic
        import on the RUNTIME_COMPILED_PROTO_MODULE_DIR's environment variable.

        Args:
            config(aconfig.Config): caikit configuration

        Returns:
            (module): Handle to the module after dynamic import
        """
        if not config:
            config = get_config()
        module_dir = config.runtime.compiled_proto_module_dir
        service_proto_gen_module = import_util.get_dynamic_module(module, module_dir)
        if service_proto_gen_module is None:
            message = (
                "Unable to load compiled proto module: %s within dir %s"
                % (module)
                % (module_dir)
            )
            log.error("<RUN22291313E>", message)
            raise CaikitRuntimeException(grpc.StatusCode.INTERNAL, message)
        return service_proto_gen_module
