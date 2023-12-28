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
The RemoteModelInitializer loads a RemoteModuleConfig as an empty Module that
sends all requests to an external runtime server

Configuration for RemoteModelInitializer lives under the config as follows:

model_management:
    initializers:
        <initializer name>:
            type: REMOTE
"""
# Standard
from collections import OrderedDict
from contextlib import contextmanager
from functools import cached_property
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple, Type
import atexit
import copy
import json
import uuid

# Third Party
import grpc
import requests

# First Party
import aconfig
import alog

# Local
from caikit.config.config import merge_configs
from caikit.core.data_model import DataBase, DataStream
from caikit.core.exceptions import error_handler
from caikit.core.exceptions.caikit_core_exception import (
    CaikitCoreException,
    CaikitCoreStatusCode,
)
from caikit.core.model_management.model_initializer_base import ModelInitializerBase
from caikit.core.modules import ModuleBase, module
from caikit.core.modules.remote_config import RemoteModuleConfig
from caikit.core.task import TaskBase
from caikit.interfaces.runtime.server import (
    MODEL_ID,
    OPTIONAL_INPUTS_KEY,
    REQUIRED_INPUTS_KEY,
    get_grpc_route_name,
    get_http_route_name,
)
from caikit.interfaces.runtime.service import (
    MODEL_MESH_MODEL_ID_KEY,
    CaikitRPCDescriptor,
    ServiceType,
)

log = alog.use_channel("RINIT")
error = error_handler.get(log)


class RemoteModelInitializer(ModelInitializerBase):
    __doc__ = __doc__
    name = "REMOTE"

    def __init__(self, config: aconfig.Config, instance_name: str):
        """Construct with the config"""
        self._instance_name = instance_name
        self._module_class_map = {}

    def init(self, model_config: RemoteModuleConfig, **kwargs) -> Optional[ModuleBase]:
        """Given a RemoteModuleConfig, initialize a RemoteModule instance"""

        # Ensure the module config was produced by a RemoteModelFinder
        error.type_check(
            "<COR47750753E>", RemoteModuleConfig, model_config=model_config
        )

        # Construct remote module class if one has not already been created
        if model_config.module_id not in self._module_class_map:
            self._module_class_map[
                model_config.module_id
            ] = self.construct_module_class(model_config)

        remote_module_class = self._module_class_map[model_config.module_id]
        return remote_module_class(model_config.connection, model_config.model_path)

    def construct_module_class(
        self, model_config: RemoteModuleConfig
    ) -> Type[ModuleBase]:
        """Helper function to construct a ModuleClass. This is a separate function to allow
         for easy overloading

         Args:
             model_config: RemoteModuleConfig
                The model config to construct the module from

        Returns:
            module: Type[ModuleBase]
                The constructed module"""
        return construct_remote_module_class(model_config)


class _RemoteModelBaseClass(ModuleBase):
    """Private class to act as the base for remote modules. This class will be subclassed and mutated by
    construct_remote_module_class to make it have the same functions and parameters as the source module."""

    def __init__(self, connection_info: Dict[str, Any], model_name: str):
        # Initialize module base
        super().__init__()

        self._model_name = model_name
        self._connection = connection_info

        # Load connection parameters and assert types
        self._hostname = self._connection.get("hostname")
        self._port = self._connection.get("port")
        self._protocol = self._connection.get("protocol")
        self._timeout = self._connection.get("timeout")
        self._options = self._connection.get("options", {})
        self._model_key = self._connection.get("model_key", MODEL_MESH_MODEL_ID_KEY)

        error.value_check(
            "<COR72284545E>",
            self._protocol in ["grpc", "http"],
            "Unknown protocol: %s",
            self._protocol,
        )
        error.type_check(
            "<COR73221567E>", str, int, hostname=self._hostname, port=self._port
        )
        error.type_check(
            "<COR7331567E>",
            int,
            allow_none=True,
            timeout=self._timeout,
        )

        # Load TLS files on startup to stop unneeded reads
        tls_info = self._connection.get("tls", {})
        self._tls_enabled = tls_info.get("enabled", False)
        self._tls_insecure_verify = tls_info.get("insecure_verify", False)

        # Gather CA file info and data
        self._ca_data = None
        self._ca_file_name = tls_info.get("ca_file")
        if self._ca_file_name:
            ca_file = Path(self._ca_file_name)
            if not ca_file.exists():
                raise FileNotFoundError(
                    f"Unable to find TLS CA file at {self._ca_file_name}"
                )
            self._ca_data = ca_file.read_bytes()

        # Gather MTLS Cert
        self._mtls_cert_data = None
        self._mtls_cert_file_name = tls_info.get("cert_file")
        if self._mtls_cert_file_name:
            mtls_cert_file = Path(self._mtls_cert_file_name)
            if not mtls_cert_file.exists():
                raise FileNotFoundError(
                    f"Unable to find TLS CA file at {self._mtls_cert_file_name}"
                )
            self._mtls_cert_data = mtls_cert_file.read_bytes()

        # Gather MTLS Key
        self._mtls_key_data = None
        self._mtls_key_file_name = tls_info.get("key_file")
        if self._mtls_key_file_name:
            mtls_cert_file = Path(self._mtls_key_file_name)
            if not mtls_cert_file.exists():
                raise FileNotFoundError(
                    f"Unable to find TLS CA file at {self._mtls_key_file_name}"
                )
            self._mtls_key_data = mtls_cert_file.read_bytes()

        # Assert inputs are valid types
        error.type_check(
            "<COR74343245E>",
            bool,
            tls_enabled=self._tls_enabled,
            tls_insecure_verify=self._tls_insecure_verify,
        )
        error.type_check(
            "<COR74353245E>",
            str,
            allow_none=True,
            mtls_key_file_name=self._mtls_key_file_name,
            mtls_cert_file_name=self._mtls_cert_file_name,
        )
        error.type_check(
            "<COR74352245E>",
            bytes,
            allow_none=True,
            mtls_key_data=self._mtls_key_data,
            mtls_cert_data=self._mtls_cert_data,
        )

        if self._protocol == "grpc" and self._tls_enabled:
            error.value_check(
                "<COR74252245E>",
                not self._tls_insecure_verify,
                "GRPC does not support insecure TLS connections. Please provide a valid CA certificate",
            )

    ### Method Factories

    @classmethod
    def generate_train_function(cls, method: CaikitRPCDescriptor) -> Callable:
        """Factory function to construct a train function that will then be set as an attribute"""

        def train_func(self, *args, **kwargs) -> method.signature.return_type:
            train_kwargs = {}
            if "_output_path" in kwargs:
                train_kwargs["output_path"] = kwargs.pop("_output_path")
            train_kwargs["model_name"] = kwargs.pop(
                "_model_name", f"{self._model_name}-{uuid.uuid4()}"
            )

            bound_args = method.signature.method_signature.bind(*args, **kwargs)
            train_kwargs["parameters"] = DataBase.get_class_for_name(
                method.request_dm_name.replace("Request", "Parameters")
            )(**bound_args.arguments)

            # Set return type to TrainType
            method.response_dm_name = "TrainingJob"
            training_response = self.remote_method_request(
                method, ServiceType.TRAINING, **train_kwargs
            )
            return cls(self._connection, training_response.model_name)

        # Override infer function name attributes and signature
        train_func.__name__ = method.signature.method_name
        train_func.__qualname__ = method.signature.qualified_name
        train_func.__signature__ = method.signature.method_signature
        return train_func

    @classmethod
    def generate_inference_function(
        cls, task: Type[TaskBase], method: CaikitRPCDescriptor
    ) -> Callable:
        """Factory function to construct inference functions that will be set as an attribute."""

        def infer_func(self, *args, **kwargs) -> method.signature.return_type:
            return self.remote_method_request(
                method,
                ServiceType.INFERENCE,
                *args,
                **kwargs,
            )

        # Override infer function name attributes and signature
        infer_func.__name__ = method.signature.method_name
        infer_func.__qualname__ = method.signature.qualified_name
        infer_func.__signature__ = method.signature.method_signature

        # Wrap infer function with task method to ensure internal attributes are properly
        # set
        task_wrapped_infer_func = task.taskmethod(
            method.input_streaming, method.output_streaming
        )(infer_func)
        return task_wrapped_infer_func

    ### Remote Interface

    def remote_method_request(
        self, method: CaikitRPCDescriptor, service_type: ServiceType, *args, **kwargs
    ) -> Any:
        """Function to run a remote request based on the data stored in CaikitRPCDescriptor"""
        if self._protocol == "grpc":
            return self._request_via_grpc(method, service_type, *args, **kwargs)
        elif self._protocol == "http":
            return self._request_via_http(method, service_type, *args, **kwargs)

        raise NotImplementedError(f"Unknown protocol {self._protocol}")

    ### HTTP Helper Functions
    def _request_via_http(
        self,
        method: CaikitRPCDescriptor,
        service_type: ServiceType,
        *args,
        **kwargs,
    ) -> Any:
        # Get request data model
        request_dm = DataBase.get_class_for_name(method.request_dm_name)(
            *args, **kwargs
        )

        # ! This is a hack to ensure all fields/types have been json encoded (bytes/datetime/etc)
        request_dm_dict = json.loads(request_dm.to_json(use_oneof=True))

        # Parse generic Request type into HttpRequest format
        if service_type == ServiceType.INFERENCE:
            http_request_dict = {
                REQUIRED_INPUTS_KEY: {},
                OPTIONAL_INPUTS_KEY: {},
                MODEL_ID: self._model_name,
            }
            for param in method.signature.parameters:
                value = request_dm_dict.get(param)

                # If param doesn't have a default then add it to inputs
                if param not in method.signature.default_parameters:
                    http_request_dict[REQUIRED_INPUTS_KEY][param] = value

                # If the param is different then the default then add it to parameters
                elif value != method.signature.default_parameters.get(param):
                    http_request_dict[OPTIONAL_INPUTS_KEY][param] = value

            # If there is only one input then collapse down the value
            if len(http_request_dict[REQUIRED_INPUTS_KEY]) == 1:
                http_request_dict[REQUIRED_INPUTS_KEY] = list(
                    http_request_dict[REQUIRED_INPUTS_KEY].values()
                )[0]
        elif service_type == ServiceType.TRAINING:
            # Strip all null values
            def _remove_null_values(_attr):
                if isinstance(_attr, dict):
                    return {
                        key: _remove_null_values(value)
                        for key, value in _attr.items()
                        if value
                    }
                if isinstance(_attr, list):
                    return [
                        _remove_null_values(listitem) for listitem in _attr if listitem
                    ]

                return _attr

            http_request_dict = _remove_null_values(request_dm_dict)

        # Get request options and target
        request_kwargs = {"headers": {"Content-type": "application/json"}}
        if self._tls_enabled:
            if self._mtls_cert_file_name and self._mtls_key_file_name:
                request_kwargs["cert"] = (
                    self._mtls_cert_file_name,
                    self._mtls_key_file_name,
                )

            if self._tls_insecure_verify:
                request_kwargs["verify"] = False
            else:
                request_kwargs["verify"] = self._ca_file_name or True

        if method.output_streaming:
            request_kwargs["stream"] = True

        if self._timeout:
            request_kwargs["timeout"] = self._timeout

        request_kwargs = merge_configs(request_kwargs, self._options)

        request_url = (
            f"{self._get_remote_target()}{get_http_route_name(method.rpc_name)}"
        )

        # Send request
        response = requests.post(request_url, json=http_request_dict, **request_kwargs)
        if response.status_code != 200:
            raise CaikitCoreException(
                CaikitCoreStatusCode.UNKNOWN,
                f"Received status {response.status_code} from remote server: {response.text}",
            )

        # Parse response data model either as file or json
        response_dm_class = DataBase.get_class_for_name(method.response_dm_name)

        if method.output_streaming:

            def stream_parser():
                """Helper Generator to parse SSE events"""
                for line in response.iter_lines():
                    # Skip empty or event lines as they're constant
                    if not line or b"event" in line:
                        continue

                    # Split data lines and remove data: tags before parsing by DM
                    decoded_response = line.decode(response.encoding).replace(
                        "data: ", ""
                    )
                    yield response_dm_class.from_json(decoded_response)

            return DataStream(stream_parser)

        if response_dm_class.supports_file_operations:
            return response_dm_class.from_file(response.text)

        return response_dm_class.from_json(response.text)

    ### GRPC Helper Functions

    def _request_via_grpc(
        self,
        method: CaikitRPCDescriptor,
        service_type: ServiceType,
        *args,
        **kwargs,
    ) -> Any:
        """Helper function to send a grpc request"""

        # Get the request types
        request_dm_class = DataBase.get_class_for_name(method.request_dm_name)
        request_protobuf_class = request_dm_class.get_proto_class()

        # Get the response types
        response_dm_class = DataBase.get_class_for_name(method.response_dm_name)
        response_protobuf_class = response_dm_class.get_proto_class()

        # Get the RPC route
        grpc_route = get_grpc_route_name(service_type, method.rpc_name)

        # Construct the service_rpc and serializers
        if method.input_streaming and method.output_streaming:
            service_rpc = self._grpc_channel.stream_stream(
                grpc_route,
                request_serializer=request_protobuf_class.SerializeToString,
                response_deserializer=response_protobuf_class.FromString,
            )
        elif method.input_streaming:
            service_rpc = self._grpc_channel.stream_unary(
                grpc_route,
                request_serializer=request_protobuf_class.SerializeToString,
                response_deserializer=response_protobuf_class.FromString,
            )
        elif method.output_streaming:
            service_rpc = self._grpc_channel.unary_stream(
                grpc_route,
                request_serializer=request_protobuf_class.SerializeToString,
                response_deserializer=response_protobuf_class.FromString,
            )
        else:
            service_rpc = self._grpc_channel.unary_unary(
                grpc_route,
                request_serializer=request_protobuf_class.SerializeToString,
                response_deserializer=response_protobuf_class.FromString,
            )

        # Construct request object
        if method.input_streaming:
            # Bind the args and kwargs to the signature for parsing. Use None for the self argument
            bound_args = method.signature.method_signature.bind(None, *args, **kwargs)
            bound_args.arguments.pop("self")

            # Gather all iterable parameters as these should be streamed
            streaming_kwargs = OrderedDict()
            for name in self._get_streaming_arguments(**bound_args.arguments):
                streaming_kwargs[name] = bound_args.arguments.pop(name)

            def input_stream_parser():
                """Helper function to iterate over a datastream and stream requests"""
                for stream_tuple in DataStream.zip(*streaming_kwargs.values()):
                    stream_arguments = copy.deepcopy(bound_args)
                    for streaming_key, sub_value in zip(
                        streaming_kwargs.keys(), stream_tuple
                    ):
                        stream_arguments.arguments[streaming_key] = sub_value

                    yield request_dm_class(
                        *stream_arguments.args, **stream_arguments.kwargs
                    ).to_proto()

            grpc_request = input_stream_parser()
        else:
            grpc_request = request_dm_class(*args, **kwargs).to_proto()

        # Send RPC request with or without streaming
        if method.output_streaming:

            def output_stream_parser():
                """Helper function to stream result objects"""
                for proto in service_rpc(
                    grpc_request,
                    metadata=[(self._model_key, self._model_name)],
                    timeout=self._timeout,
                ):
                    yield response_dm_class.from_proto(proto)

            return DataStream(output_stream_parser)
        else:
            response = service_rpc(
                grpc_request,
                metadata=[(self._model_key, self._model_name)],
                timeout=self._timeout,
            )
            return response_dm_class.from_proto(response)

    @cached_property
    def _grpc_channel(self) -> grpc.Channel:
        """Helper function to construct a GRPC channel
        with correct credentials and TLS settings."""
        # Gather grpc configuration
        target = self._get_remote_target()
        options = list(self._options.items())

        # Generate secure channel
        if self._tls_enabled:
            grpc_credentials = grpc.ssl_channel_credentials(
                root_certificates=self._ca_data,
                private_key=self._mtls_key_data,
                certificate_chain=self._mtls_cert_data,
            )
            channel = grpc.secure_channel(
                target, credentials=grpc_credentials, options=options
            )
        else:
            channel = grpc.insecure_channel(target, options=options)

        # Use atexit to ensure Channel is closed before process termination.
        # atexit is required instead of __del__ as streaming methods might still be using this Channel
        # after all references to the Module have been lost.
        atexit.register(_RemoteModelBaseClass._close_grpc_channel, channel)

        return channel

    @staticmethod
    def _close_grpc_channel(channel: grpc.Channel):
        """Helper function to close a grpc channel. This should
        be call at program exit."""
        channel.close()

    ### Generic Helper Functions

    def _get_remote_target(self) -> str:
        """Get the current remote target"""
        target_string = f"{self._hostname}:{self._port}"
        if self._protocol == "grpc":
            return target_string
        else:
            if self._tls_enabled:
                return f"https://{target_string}"
            else:
                return f"http://{target_string}"

    @staticmethod
    def _get_streaming_arguments(**kwargs: Dict[str, Any]) -> List[str]:
        """Helper function to detect which kwargs are streaming"""
        streaming_arguments = []
        for name, value in kwargs.items():
            if isinstance(value, (DataStream, Generator)):
                streaming_arguments.append(name)
        return streaming_arguments


def construct_remote_module_class(
    model_config: RemoteModuleConfig,
    model_class: Type[_RemoteModelBaseClass] = _RemoteModelBaseClass,
) -> Type[ModuleBase]:
    """Factory function to construct unique Remote Module Class."""

    # Construct unique class which will have functions attached to it
    RemoteModelClass: Type[_RemoteModelBaseClass] = type(
        "RemoteModelClass", (model_class,), {}
    )

    # Add the method signatures for train and each task
    if model_config.train_method:
        train_func = RemoteModelClass.generate_train_function(model_config.train_method)
        setattr(
            RemoteModelClass,
            model_config.train_method.signature.method_name,
            train_func,
        )

    task_list = []
    for task, method_descriptions in model_config.task_methods:
        task_list.append(task)
        for description in method_descriptions:
            func = RemoteModelClass.generate_inference_function(task, description)
            setattr(RemoteModelClass, description.signature.method_name, func)

    # Wrap Module with decorator to ensure attributes are properly set
    RemoteModelClass = module(
        id=model_config.module_id,
        name=model_config.module_name,
        version="0.0.0",
        tasks=task_list,
        # We should make a remote backend that just stores signatures
        backend_type="LOCAL",
    )(RemoteModelClass)

    return RemoteModelClass
