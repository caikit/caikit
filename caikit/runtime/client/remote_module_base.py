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
The RemoteModuleBase is a base class that can be mutated to have the same task methods
as a ModuleBase but submit requests to a remote runtime instead of loading locally. By 
design this class/factory does not use any references to the original Module class.
"""
# Standard
from collections import OrderedDict
from threading import Lock
from typing import Any, Callable, Dict, Generator, List, Type, Union
import copy
import inspect
import json
import uuid

# Third Party
from requests import HTTPError, RequestException, Session
import grpc

# First Party
import alog

# Local
from caikit.core.data_model import DataBase, DataStream
from caikit.core.exceptions import error_handler
from caikit.core.modules import ModuleBase, module
from caikit.core.task import TaskBase
from caikit.interfaces.common.data_model import ConnectionInfo, Sequence
from caikit.runtime.client.remote_config import RemoteModuleConfig, RemoteRPCDescriptor
from caikit.runtime.client.utils import (
    construct_grpc_channel,
    construct_requests_session,
)
from caikit.runtime.names import (
    HTTP_TO_STATUS_CODE,
    MODEL_ID,
    OPTIONAL_INPUTS_KEY,
    REQUIRED_INPUTS_KEY,
    ServiceType,
    get_grpc_route_name,
    get_http_route_name,
)
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException

log = alog.use_channel("RMBASE")
error = error_handler.get(log)


class RemoteModuleBase(ModuleBase):
    """Class to act as the base for remote modules. This class will be subclassed and
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
        self._conn_channel: Union[grpc.Channel, Session] = None

        # Assert parameter values
        if self._protocol == "grpc" and self._tls.enabled:
            error.value_check(
                "<COR74451567E>",
                not self._tls.insecure_verify,
                "GRPC does not support insecure TLS connections."
                "Please provide a valid CA certificate",
            )

    def __del__(self):
        """Destructor to ensure channel/session is cleaned up on deletion"""
        with self._channel_lock:
            if self._conn_channel:
                self._conn_channel.close()

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

            # 🌶️🌶️🌶️ This code martials the train function arguments/kwargs into the desired
            # TrainParameters dataobject. Use signature parsing to ensure all args are mapped to
            # the correct name. Also use string replacement as names.get_train_parameter_name
            # requires a ref to the Module
            bound_args = method.signature.method_signature.bind(*args, **kwargs)
            train_parameter_class = DataBase.get_class_for_name(
                method.request_dm_name.replace("Request", "Parameters")
            )
            train_kwargs["parameters"] = train_parameter_class(**bound_args.arguments)

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
        request_dm_dict = json.loads(request_dm.to_json())

        # ! This is another hack to ensure all Union types match the oneOf generated by pydantic
        request_dm_dict = self._rename_union_sequence_types(
            request_dm_dict, request_dm.__class__
        )

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

        request_url = (
            f"{self._get_remote_target()}{get_http_route_name(method.rpc_name)}"
        )

        # Send request while capturing any errors and reporting them as CaikitRuntimeExceptions
        try:
            response = self._http_session.post(
                request_url, json=http_request_dict, stream=method.output_streaming
            )
        except RequestException as err:
            raise CaikitRuntimeException(
                grpc.StatusCode.UNKNOWN, "Unknown exception while connecting to runtime"
            ) from err

        if response.status_code != 200:
            # Capture any HTTP errors and return them with the proper Caikit Status mapping
            try:
                response.raise_for_status()
            except HTTPError as err:
                raise CaikitRuntimeException(
                    HTTP_TO_STATUS_CODE.get(
                        response.status_code, grpc.StatusCode.UNKNOWN
                    ),
                    f"Received status {response.status_code} from remote server: {response.text}",
                ) from err

        # Parse response data model either as file or json
        response_dm_class = DataBase.get_class_for_name(method.response_dm_name)

        if method.output_streaming:

            def stream_parser():
                """Helper Generator to parse SSE events"""
                try:
                    for line in response.iter_lines():
                        # Skip empty or event lines as they're constant
                        if "data:" in line:
                            # Split data lines and remove data: tags before parsing by DM
                            decoded_response = line.decode(response.encoding).replace(
                                "data: ", ""
                            )
                            yield response_dm_class.from_json(decoded_response)

                except RequestException as err:
                    raise CaikitRuntimeException(
                        grpc.StatusCode.UNKNOWN,
                        "Received unknown exception from remote server while streaming results",
                    ) from err

            # Attach reference of this response to the returned DataStream. This ensures
            # that requests stream won't get closed until after the DataStream has been cleaned up
            return_stream = DataStream(stream_parser)
            return_stream._source = response.content
            return return_stream

        # If the response_dm_class supports file operations than the HTTP server would've returned
        # with to_file instead of to_json. Thus for the client we need to return from_file instead
        # of from_json
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
        }
        if self._connection.timeout:
            request_kwargs["timeout"] = self._connection.timeout

        # Send RPC request with or without streaming
        if method.output_streaming:

            def output_stream_parser():
                """Helper function to stream result objects"""
                try:
                    for proto in service_rpc(grpc_request, **request_kwargs):
                        yield response_dm_class.from_proto(proto)

                except grpc.RpcError as err:
                    raise CaikitRuntimeException(
                        err.code() if hasattr(err, "code") else grpc.StatusCode.UNKNOWN,
                        "Error received while streaming GRPC result",
                    ) from err

            # Attach reference of this RemoteModuleClass to the returned DataStream. This ensures
            # the GRPC Channel won't get closed until after the DataStream has been cleaned up
            return_stream = DataStream(output_stream_parser)
            return_stream._source = self
            return return_stream
        else:
            try:
                response = service_rpc(grpc_request, **request_kwargs)
            except grpc.RpcError as err:
                raise CaikitRuntimeException(
                    err.code() if hasattr(err, "code") else grpc.StatusCode.UNKNOWN,
                    "Error received from GRPC request",
                ) from err

            return response_dm_class.from_proto(response)

    @property
    def _grpc_channel(self) -> grpc.Channel:
        """Helper function to construct a GRPC channel
        with correct credentials and TLS settings."""
        # Short circuit if channel has already been set
        if self._conn_channel:
            return self._conn_channel

        with self._channel_lock:
            # Check for the channel again incase it was created during lock acquisition
            if self._conn_channel:
                return self._conn_channel

            # Gather grpc configuration
            target = self._get_remote_target()
            options = list(self._connection.options.items())

            # Generate secure channel
            channel = construct_grpc_channel(target, options, self._tls)
            self._conn_channel = channel
            return self._conn_channel

    @property
    def _http_session(self) -> Session:
        """Helper function to construct a requests Session with
        with correct credentials and TLS settings."""
        # Short circuit if session has already been set
        if self._conn_channel:
            return self._conn_channel

        with self._channel_lock:
            # Check for the channel again incase it was created during lock acquisition
            if self._conn_channel:
                return self._conn_channel

            self._conn_channel = construct_requests_session(
                self._connection.options, self._tls, self._connection.timeout
            )
            return self._conn_channel

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

    @staticmethod
    def _rename_union_sequence_types(obj: Any, dm_type: type):
        """Helper function that renames all references in a dictionary
        to match the oneOf value of the DataModel and to collapse all Primitive
        sequences. This is required to match the format of http requests

        For example:
           {
               "union_str": "test",
               "ints": {
                   "values":[1,2,3]
               }
           }

        Becomes:
            {
                "union": "test",
                "ints":[1,2,3]
            }

        """

        if isinstance(obj, list):
            # If list contains DataObjects then recurse. Else return primitive list
            if inspect.isclass(dm_type) and issubclass(dm_type, DataBase):
                return [
                    RemoteModuleBase._rename_union_sequence_types(sub_obj, dm_type)
                    for sub_obj in obj
                ]

            return obj

        elif isinstance(obj, dict):
            # Ensure dm_type is a DataObject
            if not (inspect.isclass(dm_type) and issubclass(dm_type, DataBase)):
                raise ValueError("Dict object must map to DataBase")

            # If instance is a sequence then collapse down the values
            if inspect.isclass(dm_type) and issubclass(dm_type, Sequence):
                return obj.get("values", [])

            output_dict = {}
            for key, val in obj.items():
                # If key is apart of a Union then replace the field name with
                # the union name. E.g. data_str -> data
                dest_key = key
                if key in dm_type._fields_to_oneof:
                    dest_key = dm_type._fields_to_oneof[key]

                val_type = dm_type.get_field_message_type(key)
                output_dict[dest_key] = RemoteModuleBase._rename_union_sequence_types(
                    val, val_type
                )

            return output_dict

        # If object is a primitive then return it directly
        else:
            return obj


def construct_remote_module_class(
    model_config: RemoteModuleConfig,
    model_class: Type[RemoteModuleBase] = RemoteModuleBase,
) -> Type[ModuleBase]:
    """Factory function to construct unique Remote Module Class."""

    # Construct unique class which will have functions attached to it
    RemoteModelClass: Type[RemoteModuleBase] = type(
        "RemoteModelClass", (model_class,), dict(model_class.__dict__)
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
