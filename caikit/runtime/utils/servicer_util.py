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
"""A generic module to help Predict and Train servicers
"""
# Standard
from typing import Any, Dict, Iterable, Iterator
import traceback

# Third Party
from google.protobuf.descriptor import FieldDescriptor, ServiceDescriptor
from google.protobuf.message import Message as ProtoMessageType
import grpc

# First Party
import alog

# Local
from caikit.core.data_model import DataStream
from caikit.core.data_model.base import DataBase
from caikit.core.signature_parsing import CaikitMethodSignature
from caikit.interfaces.runtime.data_model.training_management import ModelPointer
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from caikit.runtime.utils.import_util import get_data_model

log = alog.use_channel("SERVICR-UTIL")

# Protobuf non primitives
# Ref: https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.descriptor
NON_PRIMITIVE_TYPES = [FieldDescriptor.TYPE_MESSAGE, FieldDescriptor.TYPE_ENUM]


def validate_caikit_library_class_exists(cdm, class_name):
    try:
        # Attempt to instantiate a Caikit* CDM class of the specified
        # output type
        return getattr(cdm, class_name)

    except AttributeError as e:
        log.warning("<RUN24024050W>", "No Caikit Library CDM class for %s", class_name)
        # Look up the data model class corresponding to the given name
        dm_class_name = DataBase.get_class_for_name(class_name=class_name)
        if not dm_class_name:
            log.error(
                "<RUN24024010E>", "No Caikit Library CDM class for %s", class_name
            )
            raise e
        return dm_class_name


def validate_caikit_library_class_method_exists(caikit_library_class, method_name):
    try:
        getattr(caikit_library_class, method_name)

    except AttributeError as e:
        log.error(
            "<RUN24024009E>",
            "No `%s` method for Caikit Library CDM class %s",
            method_name,
            caikit_library_class,
        )
        raise e


def build_proto_stream(
    caikit_library_response: Iterable[DataBase],
) -> Iterator[ProtoMessageType]:
    """Returns an iterator that serializes each item in the model's response to protobuf"""

    def _proto_generator():
        for item in caikit_library_response:
            try:
                yield item.to_proto()
            except Exception as e:
                log.warning(
                    {
                        "log_code": "<RUN11567943W>",
                        "message": "Exception while serializing response from stream: "
                        "{}".format(e),
                        "stack_trace": traceback.format_exc(),
                    }
                )
                raise CaikitRuntimeException(
                    grpc.StatusCode.INTERNAL,
                    "Could not serialize output in model response stream",
                ) from e

    return iter(DataStream(_proto_generator))


def build_proto_response(caikit_library_response: DataBase) -> ProtoMessageType:
    """Serializes a data model instance into a protobuf message"""
    try:
        return caikit_library_response.to_proto()
    except Exception as e:
        log.warning(
            {
                "log_code": "<RUN11230943W>",
                "message": "Exception while serializing response: {}".format(e),
                "stack_trace": traceback.format_exc(),
            }
        )
        raise CaikitRuntimeException(
            grpc.StatusCode.INTERNAL, "Could not serialize response"
        ) from e


def is_protobuf_primitive_field(obj):
    """Check whether or not a descriptor object is a protobufs primitive. The
    full list of descriptors can be found at the link below.

    https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.descriptor

    Args:
        obj (google.protobufs.pyext._message.FieldDescriptor):
            A protoc-compiled FieldDescriptor object describing a field to be
            passed as input or output in a well-structured way.
    Returns:
        boolean indicating whether or not a primitive compiled protobuf field
        was passed.
    """
    # If we have a non-primitive object
    if not isinstance(obj, FieldDescriptor):
        log.warning(
            "<RUN24033310D>",
            "Type [%s] should be FieldDescriptor",
            type(obj),
        )
        return False

    if obj.type == FieldDescriptor.TYPE_MESSAGE and obj.message_type.name == "Struct":
        return True
    return obj.type not in NON_PRIMITIVE_TYPES


def get_metadata(context, key, required=True):
    """Retrieve a value from the gRPC ServicerContext invocation metadata
    dictionary with the given key

    Args:
        context(grpc.ServicerContext): Context object (contains request metadata, etc)
        key(string): The invocation metadata dictionary key to retrieve
        required(boolean): Whether the piece of metadata is required to exist
    Returns:
        value (object): The value, or None if the specified key does not
            exist in the invocation metadata
    """
    try:
        return dict(context.invocation_metadata()).get(key, None)
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.warning(
            {
                "log_code": "<RUN15349983W>",
                "message": "Exception while attempting to extract {} from "
                "request metadata: {}".format(key, e),
                "stack_trace": traceback.format_exc(),
            }
        )
        if required:
            raise CaikitRuntimeException(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Could not read required metadata: {}".format(key),
            ) from e


def snake_to_upper_camel(string: str) -> str:
    """Simple snake -> upper camel conversion"""
    return "".join([part[0].upper() + part[1:] for part in string.split("_")])


def validate_data_model(
    service_descriptor: ServiceDescriptor,
):
    """Validate the Caikit Library Common Data Model (CDM) against a service descriptor
    that defines the RPCs that this class must support at predict/train time.
    More specifically, this function will iterate over every RPC/method
    defined in the service descriptor, and for each RPC, will validate that
    the Caikit Library CDM has classes that correspond with the field types of the
    incoming message fields, and that those Caikit Library classes have `from_proto`
    functions defined.  It will further verify that the output message
    has a corresponding Caikit Library CDM class with a `to_proto` method defined.

    Args:
        service_descriptor(google.protobuf.descriptor.ServiceDescriptor):
            A protoc-compiled ServiceDescriptor that defines the predict
            RPCs that will be serviced by this GlobalPredictServicer

    Raises:
        AttributeError if a discrepancy is found between the RPC service
        descriptor and the Caikit Library CDM, which will prevent an instance of
        this class from being instantiated
    """
    # this `cdm` was moved here from import-time
    cdm = get_data_model()
    for method in service_descriptor.methods:
        # Retrieve the descriptor of the input message for this RPC, and
        # verify that each field of the input message can be translated
        # into a corresponding object of the Caikit Library CDM, and that each
        # Caikit Library CDM class has a `from_proto` method defined
        input_proto_msg = method.input_type

        # Iterate over each field in this input RPC message...
        for field in input_proto_msg.fields:
            # and check to make that it is either a primitive protobufs type or that
            # we have a data model class that we can deserialize the protobufs with
            if not is_protobuf_primitive_field(field):

                if field.message_type and field.message_type.GetOptions().map_entry:
                    log.debug(
                        "<RUN51658878D>",
                        "Field: [%s] is a map of key type: [%s] and value type: [%s]",
                        field.name,
                        field.message_type.fields[0].type,
                        field.message_type.fields[1].type,
                    )
                    continue

                # ... or that we can get the field type name, e.g., RawDocument...
                field_type = input_proto_msg.fields_by_name[
                    field.name
                ].message_type.name

                # ...and ensuring that we can load a corresponding object from the Caikit* CDM
                caikit_library_class = validate_caikit_library_class_exists(
                    cdm, field_type
                )

                # ...and also ensuring that the Caikit Library CDM class has a `from_proto`
                # method...
                validate_caikit_library_class_method_exists(
                    caikit_library_class, "from_proto"
                )
            else:
                log.debug(
                    "<RUN51658879D>",
                    "Field: [%s] is a primitive of type: [%s]",
                    field.name,
                    field.type,
                )

        # Now we do something similar for the output RPC message, verifying
        # that we can construct an object of the Caikit Library CDM that matches
        # the specified field type, and that said Caikit Library object has a
        # to_proto method defined. No need to check for proto primitives here since
        # all Caikit library modules should return well formed "predict" messages
        # from the data model.
        output_class = method.output_type.name
        caikit_Library_class = validate_caikit_library_class_exists(cdm, output_class)
        validate_caikit_library_class_method_exists(caikit_Library_class, "to_proto")


def build_caikit_library_request_dict(
    request: ProtoMessageType,
    module_signature: CaikitMethodSignature,
) -> Dict[str, Any]:
    """Build the request kwargs dict.

    Args:
        request (ProtoMessageType):
            The request proto message to deserialize from
        module_signature (CaikitMethodSignature):
            Module signature or metadata about method on a module

    Returns:
        kwargs dict
    """
    try:
        # Request messages are data model objects so .from_proto can be used
        request_data_model_class = DataBase.get_class_for_proto(request)
        request_data_model = request_data_model_class.from_proto(request)

        # Initialize kwargs from data model fields
        kwargs_dict = request_data_model.to_kwargs()

        # 1. Remove any fields not in request
        unset_field_names = []
        for field in request.DESCRIPTOR.fields:
            try:
                if not request.HasField(field.name):
                    unset_field_names.append(field.name)
            except ValueError as e:
                log.debug2(
                    "failed to check HasField on field %s, error: %s",
                    field.name,
                    e,
                )
                # Remove empty iterables since we cannot distinguish between
                # unset and empty repeated fields
                field_value = getattr(request, field.name)
                if isinstance(field_value, Iterable) and len(field_value) == 0:
                    unset_field_names.append(field.name)
        for unset_field_name in unset_field_names:
            if unset_field_name in kwargs_dict:
                kwargs_dict.pop(unset_field_name)

        # 2. Remove any fields not in the module signature
        absent_field_names = [
            field
            for field in kwargs_dict.keys()
            if field not in module_signature.parameters.keys()
        ]
        for absent_field_name in absent_field_names:
            kwargs_dict.pop(absent_field_name)

        # 3. Model Pointers
        for field_name, field_value in kwargs_dict.items():
            if isinstance(field_value, ModelPointer):
                log.debug2("field_value is a ModelPointer obj")
                model_manager = ModelManager.get_instance()
                model_retrieved = model_manager.retrieve_model(field_value.model_id)
                kwargs_dict[field_name] = model_retrieved

        log.debug2("caikit_library_request_dict returned is: %s", kwargs_dict)

        return kwargs_dict

    except CaikitRuntimeException as e:
        log.warning(
            {
                "log_code": "<RUN50530381W>",
                "message": e.message,
                "error_id": e.id,
                **e.metadata,
            }
        )
        raise e

    except Exception as e:
        log.warning(
            {
                "log_code": "<RUN15438843W>",
                "message": "Exception while deserializing request: {}".format(e),
                "stack_trace": traceback.format_exc(),
            }
        )
        raise CaikitRuntimeException(
            grpc.StatusCode.INVALID_ARGUMENT, "Could not deserialize request"
        ) from e
