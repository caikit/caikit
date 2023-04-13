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
from typing import Callable, Dict, List, Type
import dataclasses
import inspect

# Third Party
import google.protobuf.descriptor
import google.protobuf.service
import grpc

# First Party
from jtd_to_proto.json_to_service import (
    json_to_service,
    service_descriptor_to_client_stub,
    service_descriptor_to_server_registration_function,
    service_descriptor_to_service,
)
import alog

# Local
from caikit.core import dataobject
from caikit.core.data_model.base import DataBase
from caikit.interfaces.runtime.data_model import (
    TrainingInfoRequest,
    TrainingInfoResponse,
)
from caikit.runtime import service_generation
from caikit.runtime.service_generation.serializers import (
    RPCSerializerBase,
    snake_to_upper_camel,
)
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from caikit.runtime.utils import import_util
from caikit.runtime.utils.config_parser import ConfigParser
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

    service: google.protobuf.service.Service
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
                compiled_pb2_package = cls._get_service_proto_module(
                    "caikit_runtime_pb2"
                )
                compiled_pb2_grpc_package = cls._get_service_proto_module(
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
                compiled_pb2_package = cls._get_service_proto_module(
                    "caikit_runtime_train_pb2"
                )
                compiled_pb2_grpc_package = cls._get_service_proto_module(
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
            service_descriptor = json_to_service(
                name=TRAINING_MANAGEMENT_SERVICE_NAME,
                package="caikit.runtime.training",
                json_service_def=TRAINING_MANAGEMENT_SERVICE_SPEC,
            )

            return ServicePackage(
                service=service_descriptor_to_service(service_descriptor),
                descriptor=service_descriptor,
                registration_function=service_descriptor_to_server_registration_function(
                    service_descriptor
                ),
                stub_class=service_descriptor_to_client_stub(service_descriptor),
                messages=None,  # we don't need messages here
            )

        if source == cls.ServiceSource.GENERATED:
            # First make sure we import the data model for the correct library
            # !!!! This will use the `caikit_library` config
            _ = import_util.get_data_model()

            lib = ConfigParser.get_instance().caikit_library
            ai_domain_name = snake_to_upper_camel(lib.replace("caikit_", ""))
            package_name = f"caikit.runtime.{ai_domain_name}"

            # Then do API introspection to come up with all the API definitions to support
            modules = [
                module_class
                for module_class in caikit.core.MODULE_REGISTRY.values()
                if module_class.__module__.partition(".")[0] == lib
            ]
            if service_type == cls.ServiceType.INFERENCE:
                task_rpc_list = service_generation.create_inference_rpcs(modules)
                service_name = f"{ai_domain_name}Service"
            else:  # service_type == cls.ServiceType.TRAINING
                task_rpc_list = service_generation.create_training_rpcs(modules)
                service_name = f"{ai_domain_name}TrainingService"

            for rpc in task_rpc_list:
                if rpc.return_type is None:
                    # TODO: need to hook up the excluded tasks / modules configs and add some
                    # more handling here to ensure good RPCs generated
                    log.info("Skipping rpc %s, no return type on method!", rpc.name)
            task_rpc_list = [
                rpc for rpc in task_rpc_list if rpc.return_type is not None
            ]

            request_data_models = cls._create_request_message_types(
                task_rpc_list, package_name
            )

            client_module = ModuleType(
                "ClientMessages",
                "Package with service message class implementations",
            )

            for dm_class in request_data_models:
                # We need the message class that data model serializes to
                setattr(client_module, dm_class.__name__, type(dm_class().to_proto()))

            service_json = cls._create_service_json(task_rpc_list, package_name)
            service_descriptor = json_to_service(
                name=service_name, package=package_name, json_service_def=service_json
            )

            return ServicePackage(
                service=service_descriptor_to_service(service_descriptor),
                descriptor=service_descriptor,
                registration_function=service_descriptor_to_server_registration_function(
                    service_descriptor
                ),
                stub_class=service_descriptor_to_client_stub(service_descriptor),
                messages=client_module,
            )

    # Implementation details for pure python service packages #
    @staticmethod
    def _create_request_message_types(
        rpcs_list: List[RPCSerializerBase],
        package_name: str,
    ) -> List[Type[DataBase]]:
        """Dynamically create data model classes for the inputs to these RPCs"""
        data_model_classes = []
        for task in rpcs_list:
            schema = {
                # triple e.g. ('caikit.interfaces.common.ProducerPriority', 'producer_id', 1)
                # This does not take care of nested descriptors
                triple[1]: triple[0]
                for triple in task.request.triples
            }

            if not schema:
                # hacky hack hack: make sure we actually have a schema to generate
                continue

            decorator = dataobject(
                schema=schema,
                package=package_name,
                optional_property_names=task.request.default_set,
            )
            cls_ = type(task.request.name, (object,), {})
            decorated_cls = decorator(cls_)
            data_model_classes.append(decorated_cls)

        return data_model_classes

    @staticmethod
    def _create_service_json(
        rpcs_list: List[RPCSerializerBase], package_name: str
    ) -> Dict:
        """Make a json service def out of some rpc defs"""
        rpc_jsons = []
        for task in rpcs_list:
            # The return type should be a "data model" object.
            # We can take advantage of the fact that all of these should contain a private
            # `proto_class` field ...and use that to yoink the fully qualified name of the
            # descriptor
            output_type_name = task.return_type.get_proto_class().DESCRIPTOR.full_name

            rpc_jsons.append(
                {
                    "name": f"{task.name}",
                    "input_type": f"{package_name}.{task.request.name}",
                    "output_type": output_type_name,
                }
            )
        service_json = {"service": {"rpcs": rpc_jsons}}
        return service_json

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
    ) -> google.protobuf.service.Service:
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
        config_parser = ConfigParser.get_instance()
        lib_names = import_util.clean_lib_names(config_parser.caikit_library)
        assert len(lib_names) == 1, "Only 1 caikit library supported for now"
        return ServicePackageFactory._snake_to_upper_camel(
            lib_names[0].replace("caikit_", "")
        )

    @staticmethod
    def _get_service_proto_module(
        module: str,
        config: ConfigParser = None,
    ) -> ModuleType:
        """
        Dynamically import the service module. This is accomplished via dynamic
        import on the SERVICE_PROTO_GEN_MODULE_DIR's environment variable.

        Args:
            config(ConfigParser): Config parser instance

        Returns:
            (module): Handle to the module after dynamic import
        """
        if not config:
            config = ConfigParser.get_instance()
        module_dir = config.service_proto_gen_module_dir
        service_proto_gen_module = import_util.get_dynamic_module(module, module_dir)
        if service_proto_gen_module is None:
            message = (
                "Unable to load service proto gen module: %s within dir %s"
                % (module)
                % (module_dir)
            )
            log.error("<RUN22291313E>", message)
            raise CaikitRuntimeException(grpc.StatusCode.INTERNAL, message)
        return service_proto_gen_module

    @staticmethod
    def _snake_to_upper_camel(string: str) -> str:
        """Simple snake -> upper camel conversion"""
        return "".join([part[0].upper() + part[1:] for part in string.split("_")])
