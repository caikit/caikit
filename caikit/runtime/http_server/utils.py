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
This module holds utility functions and classes used only by the  REST server,
this includes things like parameter handles and openapi spec generation
"""
# Standard
from typing import Any, Dict, Optional

# Local
from ...config.config import merge_configs


def convert_json_schema_to_multipart(json_schema, defs):
    """Helper function to convert a json schema from applicaiton/json into one
    that can be used for multipart requests"""
    sparse_schema, extracted_files = _extract_raw_from_schema(json_schema, defs)
    sparse_schema["properties"] = {
        **sparse_schema.get("properties", {}),
        **extracted_files,
    }
    return sparse_schema


def _extract_raw_from_schema(
    json_schema: Any, defs: Dict[str, Any], current_path=None
) -> (dict, dict):
    """Helper function to extract all "bytes" or File fields from a json schema and return the
    cleaned schema dict and a dict of extracted schemas where the key is the original raw's path"""
    if isinstance(json_schema, dict):
        # If this json_schema represents a raw field extract it
        if raw_json_schema := _parse_raw_json_schema(json_schema):
            return None, {_clean_schema_path(current_path): raw_json_schema}

        # If this json_schema is just a ref then just recurse on the ref's json to
        # extract the file information. However, don't modify the original json
        # ref schema
        if "$ref" in json_schema:
            # Fetch ref json
            local_ref_name = json_schema["$ref"].replace("#/$defs/", "")
            sub_json_obj = defs.get(local_ref_name)
            # Extract files
            _, extracted_bytes = _extract_raw_from_schema(
                sub_json_obj, defs, current_path
            )
            # Return original ref schema and file info
            return json_schema, extracted_bytes

        # If this is a generic schema then recurse on it
        output_schema = {}
        extracted_schemas = {}
        for key in json_schema:
            # format sub path
            key_path = key
            if current_path:
                key_path = current_path + "." + key

            # Recurse on schemas
            updated_schema, extracted_bytes = _extract_raw_from_schema(
                json_schema[key], defs, key_path
            )
            if updated_schema:
                output_schema[key] = updated_schema

            extracted_schemas = {**extracted_schemas, **extracted_bytes}

        return output_schema, extracted_schemas

    # If schema is a list then recurse on each sub item
    if isinstance(json_schema, list):
        output_schema = []
        extracted_schemas = {}
        for schema in json_schema:
            # Recurse on sub schema with the same path
            updated_schema, extracted_bytes = _extract_raw_from_schema(
                schema, defs, current_path
            )
            if updated_schema:
                output_schema.append(updated_schema)

            extracted_schemas = {**extracted_schemas, **extracted_bytes}
        return output_schema, extracted_schemas

    # If schema is a raw type then just return it
    return json_schema, {}


def _clean_schema_path(path):
    """Clean a schema path of all reserved openapi fields. For example this turns
    inputs.properties.anyOf.file.properties.filename  to inputs.file.filename"""
    cleared_path = (
        path.replace("allOf", "")
        .replace("anyOf", "")
        .replace("oneOf", "")
        .replace("additionalProperties", "")
        .replace("properties", "")
        .replace("items", "")
    )
    cleared_path_split = cleared_path.split(".")
    cleared_path_removed = [x for x in cleared_path_split if x]
    return ".".join(cleared_path_removed)


def _parse_raw_json_schema(json_schema: dict) -> Optional[dict]:
    """Helper to check if a json schema matches a raw objects schema. If it does return the generic
    binary openapi schema"""
    generic_binary_schema = {"type": "string", "format": "binary"}

    # If schema matches raw bytes
    if json_schema.get("type") == generic_binary_schema.get("type") and json_schema.get(
        "format"
    ) == generic_binary_schema.get("format"):
        return json_schema

    # If schema matches list of bytes
    if (
        json_schema.get("type") == "array"
        and json_schema.get("items", {}).get("type")
        == generic_binary_schema.get("type")
        and json_schema.get("items", {}).get("format")
        == generic_binary_schema.get("format")
    ):
        return json_schema

    # If schema matches a file reference then return the generic bytes schema
    if json_schema.get("title") in ["caikit_data_model.common.File"]:
        json_schema = {**json_schema, **generic_binary_schema}
        json_schema.pop("properties", None)
        return json_schema
    # If schema is a list of file references
    if json_schema.get("type") == "array" and json_schema.get("items", {}).get(
        "title"
    ) in ["caikit_data_model.common.File"]:
        json_schema["items"] = generic_binary_schema
        return json_schema

    return None


def flatten_json_schema(json_schema: dict) -> dict:
    """Function to flatten a json schema. It replaces all references to $def
    with the requested object or {} if it's not found"""
    # Remove left over $defs field
    refs_map = {"$defs": json_schema.get("$defs", {})}

    # Replace refs and remove the defs object. Don't do this to
    # json_schema to not affect the source dict
    flattened_schema = _replace_json_refs(json_schema, refs_map)
    flattened_schema.pop("$defs")
    return flattened_schema


def _replace_json_refs(current_json: Any, refs_map: dict):
    """Helper function to replace all items of {'$ref':'#/<refs>'} with the raw
    objects. This is used for generating flattened openapi specs"""

    # If object is dict than check for ref keys
    if isinstance(current_json, dict):
        if "$ref" in current_json:
            ref_key_list = current_json["$ref"].split("/")

            # find ref object, ignoring the first object as it's always
            # '#'/
            current_place = refs_map
            for key in ref_key_list[1:]:
                current_place = current_place.get(key, {})

            return _replace_json_refs(current_place, refs_map)

        # If not $ref then recurse
        return {
            key: _replace_json_refs(value, refs_map)
            for key, value in current_json.items()
        }

    # If object is list than recurse on each item
    if isinstance(current_json, list):
        return [_replace_json_refs(item, refs_map) for item in current_json]

    # If object is other type than return raw object
    return current_json


def update_dict_at_dot_path(dict_obj: dict, key: str, updated_value: Any) -> bool:
    """Helper to set values in a dict using 'foo.bar' key notation

    Args:
        dict_obj:  dict
            The dict into which the key will be set
        key:  str
            Key that may contain '.' notation indicating dict nesting
        updated_value:  Any
            The value to place at the nested key

    Returns:
        bool:
            Weather the dict was successfully updated
    """
    parts = key.split(".")
    for part in parts[:-1]:
        dict_obj = dict_obj.setdefault(part, {})
        if not isinstance(dict_obj, dict):
            return False

    # If value already exists and is a dict and the target value is a dict then
    # deep merge the keys
    if isinstance(dict_obj.get(parts[-1]), dict) and isinstance(updated_value, dict):
        merge_configs(dict_obj[parts[-1]], updated_value)
    else:
        dict_obj[parts[-1]] = updated_value
    return True
