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
This module holds the Pydantic wrapping required by the REST server,
capable of converting to and from Pydantic models to our DataObjects.
"""
# Standard
from datetime import date, datetime, time, timedelta
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    Union,
    cast,
    get_args,
    get_type_hints,
    get_origin,
)
import base64
import enum
import inspect
import json
from dataclasses import MISSING, Field as DataclassField
from typing_extensions import Doc

# Third Party
from fastapi import Request, status
from fastapi.datastructures import FormData
from fastapi.exceptions import HTTPException, RequestValidationError
from pydantic.fields import FieldInfo
from pydantic.functional_validators import BeforeValidator
from starlette.datastructures import UploadFile
import numpy as np
import pydantic

# First Party
from py_to_proto.dataclass_to_proto import (  # Imported here for 3.8 compat
    Annotated,
    get_origin,
)
import alog

# Local
from caikit.core.data_model.base import DataBase
from caikit.core.data_model.dataobject import Hidden
from caikit.interfaces.common.data_model import File
from caikit.interfaces.common.data_model.primitive_sequences import (
    BoolSequence,
    FloatSequence,
    IntSequence,
    StrSequence,
)
from caikit.runtime.http_server.utils import update_dict_at_dot_path

log = alog.use_channel("SERVR-HTTP-PYDNTC")

# PYDANTIC_TO_DM_MAPPING is essentially a 2-way map of DMs <-> Pydantic models, you give it a
# pydantic model, it gives you back a DM class, you give it a
# DM class, you get back a pydantic model.
PYDANTIC_TO_DM_MAPPING: dict[Any, Any] = {
    # Map primitive sequences to lists
    StrSequence: List[str],
    IntSequence: List[int],
    FloatSequence: List[float],
    BoolSequence: List[bool],
}


# Base class for pydantic models
# We want to set the config to forbid extra attributes
# while instantiating any pydantic models
# This is done to make sure any oneofs can be
# correctly inferred by pydantic
class ParentPydanticBaseModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", protected_namespaces=())


def pydantic_to_dataobject(pydantic_model: pydantic.BaseModel) -> DataBase:
    """Convert pydantic objects to our DM objects"""
    dm_class_to_build = PYDANTIC_TO_DM_MAPPING.get(type(pydantic_model))
    assert dm_class_to_build is not None
    dm_kwargs = {}

    for field_name, field_value in pydantic_model:
        # field could be a DM:
        # pylint: disable=unidiomatic-typecheck
        if type(field_value) in PYDANTIC_TO_DM_MAPPING:
            dm_kwargs[field_name] = pydantic_to_dataobject(field_value)
        elif isinstance(field_value, list):
            if all(type(val) in PYDANTIC_TO_DM_MAPPING for val in field_value):
                dm_kwargs[field_name] = [
                    pydantic_to_dataobject(val) for val in field_value
                ]
            else:
                dm_kwargs[field_name] = field_value
        else:
            dm_kwargs[field_name] = field_value

    return dm_class_to_build(**dm_kwargs)


def dataobject_to_pydantic(dm_class: Type[DataBase]) -> Type[pydantic.BaseModel]:
    """Make a pydantic model based on the given proto message by using the data
    model class annotations to mirror as a pydantic model
    """
    # define a local namespace for type hints to get type information from.
    # This is needed for pydantic to have a handle on JsonDict and JsonDictValue while
    # creating its base model
    localns = {"JsonDict": dict, "JsonDictValue": dict}

    if dm_class in PYDANTIC_TO_DM_MAPPING:
        return PYDANTIC_TO_DM_MAPPING[dm_class]

    annotations = {
        field_name: _get_pydantic_type(field_type)
        for field_name, field_type in get_type_hints(dm_class, localns=localns).items()
    }
    pydantic_model = type(ParentPydanticBaseModel)(
        dm_class.get_proto_class().DESCRIPTOR.full_name,
        (ParentPydanticBaseModel,),
        {
            "__annotations__": annotations,
            **{
                name: None
                for name, _ in get_type_hints(
                    dm_class,
                    localns=localns,
                ).items()
            },
        },
    )

    model_changed = False
    dataclass_fields: Optional[dict[str, DataclassField]] = getattr(
        dm_class, "__dataclass_fields_stashed__", None
    )
    model_fields: Optional[dict[str, FieldInfo]] = getattr(
        pydantic_model, "model_fields", None
    )
    if dataclass_fields and model_fields is not None:
        model_changed = True
        for which_field, dataclass_field in dataclass_fields.items():
            if dataclass_field.default is not MISSING:
                model_fields[which_field].default = dataclass_field.default

            if dataclass_field.default_factory is not MISSING:
                default_factory = dataclass_field.default_factory
                # any screening that needs to be done here?
                model_fields[which_field].default_factory = default_factory

            if "description" in dataclass_field.metadata:
                description = dataclass_field.metadata.get("description")
                model_fields[which_field].description = description

            metadata = dataclass_field.metadata

            if metadata is not None:
                # Replicate some important Pydantic fields via metadata
                if "examples" in metadata:
                    model_fields[which_field].examples = metadata.get("examples")

                if "deprecated" in metadata:
                    model_fields[which_field].deprecated = metadata.get("deprecated")

                if "title" in metadata:
                    model_fields[which_field].title = metadata.get("title")

                if "repr" in metadata:
                    model_fields[which_field].repr = metadata.get("repr", True)

                if "json_schema_extra" in metadata:
                    model_fields[which_field].json_schema_extra = metadata.get(
                        "json_schema_extra"
                    )

    annotated_types = get_type_hints(dm_class, localns=localns, include_extras=True)
    for field_name, annotated in annotated_types.items():
        if get_origin(annotated) is Annotated:
            annotated_args = get_args(annotated)
            for annotated_arg in annotated_args:
                if isinstance(annotated_arg, Doc):
                    if model_fields:
                        model_fields[field_name].description = (
                            annotated_arg.documentation
                        )
                        model_changed = True
                elif isinstance(annotated_arg, Hidden):
                    model_changed = True

                    # What to do here?

    pydantic_model_base = cast(pydantic.BaseModel, pydantic_model)
    if model_changed:
        pydantic_model_base.model_rebuild(force=True)
    pydantic_model_base.__doc__ = dm_class.__doc__

    PYDANTIC_TO_DM_MAPPING[dm_class] = pydantic_model
    # also store the reverse mapping for easy retrieval
    # should be fine since we only check for dm_class in this dict
    PYDANTIC_TO_DM_MAPPING[pydantic_model] = dm_class
    assert issubclass(pydantic_model, pydantic.BaseModel)
    return pydantic_model


# pylint: disable=too-many-return-statements
def _get_pydantic_type(field_type: type) -> Union[type, Annotated]:
    """Recursive helper to get a valid pydantic type for every field type"""
    # pylint: disable=too-many-return-statements

    # Leaves: we should have primitive types and enums
    if np.issubclass_(field_type, np.integer):
        return int
    if np.issubclass_(field_type, np.floating):
        return float
    if field_type == bytes:
        return Annotated[bytes, BeforeValidator(_from_base64)]
    if field_type in (
        int,
        float,
        bool,
        str,
        dict,
        type(None),
        date,
        datetime,
        time,
        timedelta,
    ):
        return field_type
    if isinstance(field_type, type) and issubclass(field_type, enum.Enum):
        return field_type

    # These can be nested within other data models
    if (
        isinstance(field_type, type)
        and issubclass(field_type, DataBase)
        and not issubclass(field_type, pydantic.BaseModel)
    ):
        # NB: for data models we're calling the data model conversion fn
        return dataobject_to_pydantic(field_type)

    # And then all of these types can be nested in other type annotations
    if get_origin(field_type) is Annotated:
        return _get_pydantic_type(get_args(field_type)[0])
    if get_origin(field_type) is Union:
        pydantic_type = tuple(
            _get_pydantic_type(arg_type) for arg_type in get_args(field_type)
        )
        return Union[pydantic_type]  # type: ignore
    if get_origin(field_type) is list:
        return List[_get_pydantic_type(get_args(field_type)[0])]

    if get_origin(field_type) is dict:
        return Dict[
            _get_pydantic_type(get_args(field_type)[0]),
            _get_pydantic_type(get_args(field_type)[1]),
        ]

    raise TypeError(f"Cannot get pydantic type for type [{field_type}]")


def _from_base64(data: Union[bytes, str]) -> bytes:
    if isinstance(data, str):
        return base64.b64decode(data.encode("utf-8"))
    return data


async def pydantic_from_request(
    pydantic_model: Type[pydantic.BaseModel], request: Request
):
    """Function to convert a fastapi request into a given pydantic model. This
    function parses the requests Content-Type and then correctly decodes the data.
    The currently supported Content-Types are `application/json`
    and `multipart/form-data`"""
    content_type = request.headers.get("Content-Type")
    log.debug("Detected request using %s type", content_type)
    if content_type is None:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "Null media type not supported",
        )

    # If content type is json use pydantic to parse
    if content_type == "application/json":
        raw_content = await request.body()
        try:
            return pydantic_model.model_validate_json(raw_content)
        except pydantic.ValidationError as err:
            raise RequestValidationError(errors=err.errors()) from err
    # Elif content is form-data then parse the form
    elif "multipart/form-data" in content_type:
        # Get the raw form data
        raw_form = await request.form()
        return _parse_form_data_to_pydantic(pydantic_model, raw_form)
    else:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported media type: {content_type}.",
        )


def _parse_form_data_to_pydantic(
    pydantic_model: Type[pydantic.BaseModel], form_data: FormData
) -> pydantic.BaseModel:
    """Helper function to parse a fastapi form data into a pydantic model"""

    # Gather the file pydantic model. This is computed here to avoid recomputing it every loop
    pydantic_file = dataobject_to_pydantic(File)

    # Parse each form_data key into a python dict which is then
    # converted to a pydantic model via .model_validate()
    raw_model_obj = {}
    for key in form_data:
        # Get the list of objects that has the key
        # field name
        raw_objects = list(form_data.getlist(key))

        # Make sure form field actually has values
        if not raw_objects or (len(raw_objects) > 0 and not raw_objects[0]):
            log.debug4("Detected empty form key '%s'", key)
            continue

        # Get the type hint for the requested model
        sub_key_list = key.split(".")
        model_type_hints = list(_get_pydantic_subtypes(pydantic_model, sub_key_list))
        log.debug4("Gathered hints '%s' for model key '%s'", model_type_hints, key)
        if not model_type_hints:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown key '{key}'",
            )

        # Determine the root type hint if the request is a list
        is_list = False
        if get_origin(model_type_hints[0]) is list:
            is_list = True
            model_type_hints = get_args(model_type_hints[0])

        # Recheck for union incase list was a list of unions
        if get_origin(model_type_hints[0]) is Union:
            model_type_hints = get_args(model_type_hints)

        # Loop through and check for each type hint. This is required to
        # match unions. We don't have to be too specific with parsing as
        # pydantic will handle formatting
        parsed = False
        for type_hint in model_type_hints:
            # Parse any UploadFile types into the raw bytes or File information

            for n, sub_obj in enumerate(raw_objects):
                if isinstance(sub_obj, UploadFile):
                    # If we're looking for a pydantic file type then parse the
                    # structure of UploadFile. Otherwise, just return the content bytes
                    # Per type: ignore, something is completely bad here
                    if type_hint == pydantic_file:
                        raw_objects[n] = {  # type: ignore
                            "filename": sub_obj.filename,
                            "data": sub_obj.file.read(),
                            "type": sub_obj.content_type,
                        }
                    else:
                        raw_objects[n] = sub_obj.file.read()  # type: ignore

            # If type_hint is a pydantic model then parse the json
            if (
                type_hint != pydantic_file
                and inspect.isclass(type_hint)
                and issubclass(type_hint, pydantic.BaseModel)
            ):
                failed_to_parse_json = False
                for n, sub_obj in enumerate(raw_objects):
                    try:
                        raw_objects[n] = json.loads(sub_obj)  # type: ignore
                    except TypeError:
                        raise HTTPException(  # noqa: B904 # pylint: disable=raise-missing-from
                            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=f"Unable to update object at key '{key};"
                            "; expected value to be string",
                        )
                    except json.JSONDecodeError:
                        failed_to_parse_json = True
                        break

                # If the json couldn't be parsed then skip this type
                if failed_to_parse_json:
                    log.debug3("Failed to parse json for key '%s'", key)
                    continue

            # If object is not supposed to be a list then just grab the first element
            if not is_list:
                raw_objects = raw_objects[0]

            if not update_dict_at_dot_path(raw_model_obj, key, raw_objects):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Unable to update object at key '{key}'; value already exists",
                )

            # If we were able to parse the object then break out of the type loop
            parsed = True
            break

        # If the data didn't match any of the types return 422
        if not parsed:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to parse key '{key}' with types {model_type_hints}",
            )

    # Process the dict into a pydantic model before returning
    try:
        return pydantic_model.model_validate(raw_model_obj)
    except pydantic.ValidationError as err:
        raise RequestValidationError(errors=err.errors()) from err


def _get_pydantic_subtypes(
    pydantic_model: Type[pydantic.BaseModel], keys: List[str]
) -> Iterable[type]:
    """Recursive helper to get the type_hint for a field"""
    if len(keys) == 0:
        return [pydantic_model]

    # Get the type hints for the current key
    current_key = keys.pop(0)
    current_type = get_type_hints(pydantic_model).get(current_key)
    if not current_type:
        return []

    if get_origin(current_type) is Union:
        # If we're trying to capture a union then return the entire union result
        if len(keys) == 0:
            result = get_args(current_type)
            return result

        # Get the arg which matches
        for arg in get_args(current_type):
            if result := _get_pydantic_subtypes(arg, keys):
                return result
        return []

    # If object is a list then recurse on its type
    elif get_origin(current_type) is list:
        if len(keys) == 0:
            return [current_type]

        result = _get_pydantic_subtypes(get_args(current_type)[0], keys)
        return result
    else:
        result = _get_pydantic_subtypes(current_type, keys)
        return result
