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
from typing import Any, Callable, Dict, Iterable
import traceback

# Third Party
from google.protobuf.descriptor import FieldDescriptor
import google.protobuf.descriptor
import grpc

# First Party
import alog

# Local
from caikit.core.data_model.streams.data_stream import DataStream
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.service_generation.data_stream_source import get_data_stream_source
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
        log.error("<RUN24024010E>", "No Caikit Library CDM class for %s", class_name)
        raise e


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


def build_proto_response(caikit_library_response):
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
    service_descriptor: google.protobuf.descriptor.ServiceDescriptor,
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
    request, model_function: Callable
) -> Dict[str, Any]:
    """
    build the caikit_library_request_dict.

    Basically it is:

    1. Removing zero values for other primitive
    2. Removing any arg that is not supported by the model_function
    3. "from-protoing" any non-primitive fields (example - message is "from-protoed")
    """
    caikit_library_request_dict = {}
    valid_kwarg_names = set(model_function.__code__.co_varnames)
    # remove self in case it's present
    if "self" in valid_kwarg_names:
        valid_kwarg_names.remove("self")
    # remove cls in case it's present
    if "cls" in valid_kwarg_names:
        valid_kwarg_names.remove("cls")

    log.debug("valid kwarg names: %s", valid_kwarg_names)
    cdm = get_data_model()
    try:
        log.debug2(
            "We are looping though these fields: %s",
            [field.name for field in request.DESCRIPTOR.fields],
        )
        for field in request.DESCRIPTOR.fields:

            log.debug2("processing field: %s", field.name)
            #  Need to not pass in any arg that is not supported by the function
            if field.name not in valid_kwarg_names:
                continue
            field_value = getattr(request, field.name)
            if is_protobuf_primitive_field(field):
                # We don't need to convert this field to a Caikit Library CDM instance
                # We also don't set the field if it is an int 0, which happens by default
                # if a value is left empty, otherwise we would be sending values like 0 for limit
                # bool is a subclass of int (https://docs.python.org/3/library/functions.html#bool)
                # so this needs to be handled separately
                log.debug2(
                    "<RUN51658873D>",
                    "Field name [%s] with value [%s] is a primitive of type [%s]",
                    field.name,
                    field_value,
                    type(field_value),
                )
                try:
                    # optional primitive
                    if request.HasField(field.name):
                        caikit_library_request_dict[field.name] = field_value
                except ValueError as e:
                    # non-optional primitives and iterables
                    log.debug2(
                        "failed to check HasField on field %s, error: %s",
                        field.name,
                        e,
                    )
                    # iterables
                    if isinstance(field_value, Iterable):
                        if len(field_value) != 0:
                            # cast only if it's actually a list
                            if not isinstance(field_value, (str, bytes)):
                                if "training_data" in field.name:
                                    caikit_library_request_dict[
                                        field.name
                                    ] = DataStream.from_iterable(field_value)
                                else:
                                    caikit_library_request_dict[field.name] = list(
                                        field_value
                                    )
                            # if not, pass it as is. (non-optional str & bytes)
                            else:
                                caikit_library_request_dict[field.name] = field_value
                    # non-iterable primitives
                    else:
                        caikit_library_request_dict[field.name] = field_value
            else:
                log.debug2(
                    "<RUN55658873D>",
                    "field is not primitive: %s (%s) type(%s)",
                    field.name,
                    field_value,
                    type(field_value),
                )
                # iterables
                if isinstance(field_value, Iterable):
                    if len(field_value) != 0:
                        # We sure we want a DataStream? - after huddling
                        # with Joe and Travis, it seems reasonable to assume
                        # we will return a DataStream for all non-primitive
                        # iterables. Perhaps if there was guaranteed type
                        # annotation, we could be smart about it and deal with
                        # it better in the future.

                        # the list of instances to create a DataStream object
                        # with intentionally not using list comprehension for
                        # ease of reading
                        instances = []
                        for field_item in field_value:
                            # Start by getting the class name for this
                            # particular field (e.g., RawDocument)
                            class_name = type(field_item).DESCRIPTOR.name
                            # get the Caikit Library CDM class of the same name
                            caikit_library_class = getattr(cdm, class_name)
                            # Use the Caikit Library CDM class's from_proto
                            # method to turn our protobufs field message into an
                            # instance of the Caikit Library CDM class
                            instance = caikit_library_class.from_proto(field_item)
                            instances.append(instance)

                        if "training_data" in field.name:
                            data_stream = DataStream.from_iterable(instances)
                            caikit_library_request_dict[field.name] = data_stream
                        else:
                            caikit_library_request_dict[field.name] = field_value
                else:
                    log.debug2(
                        "<RUN55258876D>",
                        "field is not primitive, and also not an Iterable: %s (%s) type(%s)",
                        field.name,
                        field_value,
                        type(field_value),
                    )
                    # if it's a custom datastream dataobject
                    stream_source = get_data_stream_source(field_value)
                    if stream_source:
                        caikit_library_request_dict[
                            field.name
                        ] = stream_source.to_data_stream()

                    else:
                        log.debug2(
                            "<RUN64546176D>",
                            "field should not have stream source: %s (%s) type(%s)",
                            field.name,
                            field_value,
                            type(field_value),
                        )
                        # Start by getting the class name for this particular
                        # field (e.g., RawDocument)
                        class_name = type(field_value).DESCRIPTOR.name

                        # special case for model pointer
                        if class_name == "ModelPointer":
                            log.debug2("field_value is a ModelPointer obj")
                            if field_value.model_id:
                                model_manager = ModelManager.get_instance()
                                model_retrieved = model_manager.retrieve_model(
                                    field_value.model_id
                                )
                                caikit_library_request_dict[
                                    field.name
                                ] = model_retrieved
                        else:
                            # Now get the Caikit Library CDM class of the same
                            # name
                            caikit_library_class = getattr(cdm, class_name)
                            # Use the Caikit Library CDM class's from_proto
                            # method to turn our protobufs field message into an
                            # instance of the Caikit Library CDM class
                            instance = caikit_library_class.from_proto(field_value)
                            # Add to the request dictionary, using the message
                            # field's name as the key (since, by convention, the
                            # argument name to the module run function will be
                            # the same as the field name)
                            caikit_library_request_dict[field.name] = instance

        log.debug2(
            "caikit_library_request_dict returned is: %s", caikit_library_request_dict
        )

        return caikit_library_request_dict

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
