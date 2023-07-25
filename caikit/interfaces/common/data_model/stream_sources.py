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
This file contains interfaces required to generate DataStreamSource[T] classes
"""

# Standard
from typing import List

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber

# Local
from caikit.core.data_model import PACKAGE_COMMON, DataObjectBase, dataobject


@dataobject(PACKAGE_COMMON)
class File(DataObjectBase):
    filename: Annotated[str, FieldNumber(1)]


@dataobject(PACKAGE_COMMON)
class ListOfFiles(DataObjectBase):
    files: Annotated[List[str], FieldNumber(1)]


@dataobject(PACKAGE_COMMON)
class Directory(DataObjectBase):
    dirname: Annotated[str, FieldNumber(1)]
    extension: Annotated[str, FieldNumber(2)]


@dataobject(PACKAGE_COMMON)
class S3Files(DataObjectBase):
    # List of file paths relative to the bucket
    files: Annotated[List[str], FieldNumber(1)]

    # URI info
    endpoint: Annotated[str, FieldNumber(2)]  # begins with `http://` or `https://`
    region: Annotated[str, FieldNumber(3)]
    bucket: Annotated[str, FieldNumber(4)]

    # HMAC credentials
    accessKey: Annotated[str, FieldNumber(5)]
    secretKey: Annotated[str, FieldNumber(6)]

    # IAM credentials
    IAM_id: Annotated[str, FieldNumber(7)]
    IAM_api_ky: Annotated[str, FieldNumber(8)]

    # TLS info
    CA_bundle_key: Annotated[str, FieldNumber(9)]


# File = make_dataobject(
#             package=package,
#             proto_name=f"{cls_name}File",
#             name="File",
#             attrs={"__qualname__": f"{cls_name}.File"},
#             annotations={"filename": str},
#         )
#         ListOfFiles = make_dataobject(
#             package=package,
#             proto_name=f"{cls_name}ListOfFiles",
#             name="ListOfFiles",
#             attrs={"__qualname__": f"{cls_name}.ListOfFiles"},
#             annotations={"files": List[str]},
#         )
#         Directory = make_dataobject(
#             package=package,
#             proto_name=f"{cls_name}Directory",
#             name="Directory",
#             attrs={"__qualname__": f"{cls_name}.Directory"},
#             annotations={"dirname": str, "extension": str},
#         )
