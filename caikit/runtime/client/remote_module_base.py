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
from threading import Lock
from typing import Any, Callable, Dict, Generator, List, Type
import copy
import json
import uuid

# Third Party
import grpc
import requests

# First Party
import alog

# Local
from caikit.config.config import merge_configs
from caikit.core.data_model import DataBase, DataStream
from caikit.core.exceptions import error_handler
from caikit.core.exceptions.caikit_core_exception import (
    CaikitCoreException,
    CaikitCoreStatusCode,
)
from caikit.core.modules import ModuleBase, module
from caikit.core.task import TaskBase
from caikit.interfaces.common.data_model.remote import ConnectionInfo
from caikit.runtime.client.remote_config import RemoteModuleConfig, RemoteRPCDescriptor
from caikit.runtime.names import (
    MODEL_ID,
    OPTIONAL_INPUTS_KEY,
    REQUIRED_INPUTS_KEY,
    ServiceType,
    get_grpc_route_name,
    get_http_route_name,
)

log = alog.use_channel("RMBASE")
error = error_handler.get(log)


class RemoteModelBaseClass(ModuleBase):
    """Private class to act as the base for remote modules. This class will be subclassed and
    mutated by construct_remote_module_class to make it have the same functions and parameters
    as the source module."""

    def __init__(
        self,
        connection_info: ConnectionInfo,
        protocol: str,
        model_key: str,
        model_name: str,
    ):
        # Initialize module base
        super().__init__()

        self._model_name = model_name

        # Load connection parameters
        self._connection = connection_info
        self._tls = self._connection.tls
        self._protocol = protocol
        self._model_key = model_key

        # Configure GRPC variables and threading lock
        self._channel_lock = Lock()
        self.__grpc_channel = None

        # Assert parameter values
        if self._protocol == "grpc" and self._tls.enabled:
            error.value_check(
                "<COR74451567E>",
                not self._tls.insecure_verify,
                "GRPC does not support insecure TLS connections."
                "Please provide a valid CA certificate",
            )

    def __del__(self):
        """Destructor to ensure channel is cleaned up on deletion"""
        with self._channel_lock:
            if self.__grpc_channel:
                self._close_grpc_channel(self._grpc_channel)

    ### Method Factories

    @classmethod
    def generate_train_function(cls, method: RemoteRPCDescriptor) -> Callable:
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
            return cls(
                self._connection,
                self._protocol,
                self._model_key,
                training_response.model_name,
            )

        # Override infer function name attributes and signature
        train_func.__name__ = method.signature.method_name
        train_func.__qualname__ = method.signature.qualified_name
        train_func.__signature__ = method.signature.method_signature
        return train_func

    @classmethod
    def generate_inference_function(
        cls, task: Type[TaskBase], method: RemoteRPCDescriptor
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
        self, method: RemoteRPCDescriptor, service_type: ServiceType, *args, **kwargs
    ) -> Any:
        """Function to run a remote request based on the data stored in RemoteRPCDescriptor"""
        if self._protocol == "grpc":
            return self._request_via_grpc(method, service_type, *args, **kwargs)
        elif self._protocol == "http":
            return self._request_via_http(method, service_type, *args, **kwargs)

        raise NotImplementedError(f"Unknown protocol {self._protocol}")

    ### HTTP Helper Functions
    def _request_via_http(
        self,
        method: RemoteRPCDescriptor,
        service_type: ServiceType,
        *args,
        **kwargs,
    ) -> Any:
        # Get request data model
        request_dm = DataBase.get_class_for_name(method.request_dm_name)(
            *args, **kwargs
        )

        # ! This is a hack to ensure all fields/types have been json encoded (bytes/datetime/etc).
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
        if self._tls.enabled:
            if self._tls.cert_file and self._tls.key_file:
                request_kwargs["cert"] = (
                    self._tls.cert_file,
                    self._tls.key_file,
                )

            if self._tls.insecure_verify:
                request_kwargs["verify"] = False
            else:
                request_kwargs["verify"] = self._tls.ca_file or True

        if method.output_streaming:
            request_kwargs["stream"] = True

        if self._connection.timeout:
            request_kwargs["timeout"] = self._connection.timeout

        request_kwargs = merge_configs(
            request_kwargs, overrides=self._connection.options
        )

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

            # Attach reference of this RemoteModuleClass to the returned DataStream. This ensures
            # the GRPC Channel won't get closed until after the DataStream has been cleaned up
            return_stream = DataStream(stream_parser)
            return_stream._source = response.content
            return return_stream

        if response_dm_class.supports_file_operations:
            return response_dm_class.from_file(response.text)

        return response_dm_class.from_json(response.text)

    ### GRPC Helper Functions

    def _request_via_grpc(
        self,
        method: RemoteRPCDescriptor,
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
            # If not streaming then construct a simple request
            grpc_request = request_dm_class(*args, **kwargs).to_proto()

        request_kwargs = {
            "metadata": [(self._model_key, self._model_name)],
            "timeout": self._connection.timeout,
        }
        # Send RPC request with or without streaming
        if method.output_streaming:

            def output_stream_parser():
                """Helper function to stream result objects"""
                for proto in service_rpc(grpc_request, **request_kwargs):
                    yield response_dm_class.from_proto(proto)

            # Attach reference of this RemoteModuleClass to the returned DataStream. This ensures
            # the GRPC Channel won't get closed until after the DataStream has been cleaned up
            return_stream = DataStream(output_stream_parser)
            return_stream._source = self
            return return_stream
        else:
            response = service_rpc(grpc_request, **request_kwargs)
            return response_dm_class.from_proto(response)

    @property
    def _grpc_channel(self) -> grpc.Channel:
        """Helper function to construct a GRPC channel
        with correct credentials and TLS settings."""
        # Short circuit if channel has already been set
        if self.__grpc_channel:
            return self.__grpc_channel

        with self._channel_lock:
            # Check for the channel again incase it was created during lock acquisition
            if self.__grpc_channel:
                return self.__grpc_channel

            # Gather grpc configuration
            target = self._get_remote_target()
            options = list(self._connection.options.items())

            # Generate secure channel
            if self._tls.enabled:
                grpc_credentials = grpc.ssl_channel_credentials(
                    root_certificates=self._tls.ca_file_data,
                    private_key=self._tls.key_file_data,
                    certificate_chain=self._tls.cert_file_data,
                )
                channel = grpc.secure_channel(
                    target, credentials=grpc_credentials, options=options
                )
            else:
                channel = grpc.insecure_channel(target, options=options)

            self.__grpc_channel = channel
            return self.__grpc_channel

    @staticmethod
    def _close_grpc_channel(channel: grpc.Channel):
        """Helper function to close a grpc channel. This should
        be called on class deletion."""
        channel.close()

    ### Generic Helper Functions

    def _get_remote_target(self) -> str:
        """Get the current remote target"""
        target_string = f"{self._connection.hostname}:{self._connection.port}"
        if self._protocol == "grpc":
            return target_string
        else:
            if self._tls.enabled:
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
    model_class: Type[RemoteModelBaseClass] = RemoteModelBaseClass,
) -> Type[ModuleBase]:
    """Factory function to construct unique Remote Module Class."""

    # Construct unique class which will have functions attached to it
    RemoteModelClass: Type[RemoteModelBaseClass] = type(
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
