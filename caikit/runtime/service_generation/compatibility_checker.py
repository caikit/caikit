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
from typing import Dict

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
        return cls._instance

    @staticmethod
    def clear():
        ApiFieldNames().service_pb2_modules = []

    @staticmethod
    def get_fields_for_message(message_name: str) -> Dict[str, int]:
        """Return first matching message found for 'message_name'"""
        for service in ApiFieldNames().service_pb2_modules:
            if message_name in service.DESCRIPTOR.message_types_by_name:
                message_type = service.DESCRIPTOR.message_types_by_name[message_name]

                message_fields = message_type.fields
                return {field.name: field.number for field in message_fields}

            log.info("Did not find message name in proto module: %s", message_name)
        return {}
