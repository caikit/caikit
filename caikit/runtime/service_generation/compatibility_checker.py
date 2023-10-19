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
"""This file contains logic to ensure proto-API compatibility and model support compatibility"""

# Standard
from types import ModuleType
from typing import Dict

# Third Party
from google.protobuf import descriptor_pb2, descriptor_pool

# First Party
import alog

log = alog.use_channel("CMPAT-CHKR")


class ApiFieldNames:
    """
    Singleton class for fetching API spec from service module
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
            cls._instance.service_pb2_modules = []
            cls._instance.d_pool = None
        return cls._instance

    @staticmethod
    def add_proto_spec(
        service_pb2_module: ModuleType, d_pool: descriptor_pool.DescriptorPool
    ):
        # ApiFieldNames().service_pb2_modules.append(service_pb2_module)
        fd_proto = descriptor_pb2.FileDescriptorProto()
        service_pb2_module.DESCRIPTOR.CopyToProto(fd_proto)
        d_pool.Add(fd_proto)
        ApiFieldNames().d_pool = d_pool
        print("d_pool in add_proto_spec is: ", d_pool)

    @staticmethod
    def clear():
        ApiFieldNames().service_pb2_modules = []
        ApiFieldNames().d_pool = None

    @staticmethod
    def get_fields_for_message(message_name: str) -> Dict[str, int]:
        """Return first matching message found for 'message_name'"""
        # if len(ApiFieldNames().service_pb2_modules) == 0:
        #     log.info("There is no service modules registered with ApiFieldNames")
        # for service in ApiFieldNames().service_pb2_modules:
        if ApiFieldNames().d_pool:
            # if message_name in service.DESCRIPTOR.message_types_by_name:
            try:
                message_fd = ApiFieldNames().d_pool.FindFileByName(
                    message_name.lower() + ".proto"
                )
                message_type = message_fd.message_types_by_name[message_name]
                message_fields = message_type.fields
                fields = {field.name: field.number for field in message_fields}
                log.info(
                    "Found message %s in dpool %s from previous proto version,\
                         and it has the following fields: %s",
                    message_name,
                    ApiFieldNames().d_pool,
                    fields,
                )
                return fields

            except KeyError:
                log.info(
                    "Did not find message name %s in dpool %s with previous proto version",
                    message_name,
                    ApiFieldNames().d_pool,
                )
        return {}
