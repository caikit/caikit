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

"""Common serialization interfaces that are generally helpful for saving out models that
are not necessarily specific to any domain's 3rd party libraries.
"""
# Standard
import abc

# Local
from . import fileio


class ObjectSerializer(abc.ABC):
    """Abstract class for serializing an object to disk."""

    @abc.abstractmethod
    def serialize(self, obj, file_path):
        """Serialize the provided object to the specified file path.

        Args:
            obj (object): The object to serialize
            file_path (str): Absolute path to which the object should be
                serialized
        """


class JSONSerializer(ObjectSerializer):
    """An ObjectSerializer for serializing to a JSON file."""

    def serialize(self, obj, file_path):
        """Serialize the provided object to a JSON file.

        Args:
            obj (object): The object to serialize
            file_path (str): Absolute path to which the object should be
                serialized
        """
        fileio.save_json(obj, file_path)


class TextSerializer(ObjectSerializer):
    """An ObjectSerializer for serializing a python list to a text file."""

    def serialize(self, obj, file_path):
        """Serialize the provided python list to a text file.

        Args:
            obj (list(str)): The list to serialize
            file_path (str): Absolute path to which the object should be
                serialized
        """
        lines = "\n".join(obj)
        fileio.save_txt(lines, file_path)


class YAMLSerializer(ObjectSerializer):
    """An ObjectSerializer for serializing to a YAML file."""

    def serialize(self, obj, file_path):
        """Serialize the provided object to a YAML file.

        Args:
            obj (object): The object to serialize
            file_path (str): Absolute path to which the object should be
                serialized
        """
        fileio.save_yaml(obj, file_path)


class CSVSerializer(ObjectSerializer):
    """An ObjectSerializer for serializing to a CSV file."""

    def serialize(self, obj, file_path):
        """Serialize the provided object to a CSV file.

        Args:
            obj (object): The object to serialize
            file_path (str): Absolute path to which the object should be
                serialized
        """
        fileio.save_csv(obj, file_path)


class PickleSerializer(ObjectSerializer):
    """An ObjectSerializer for pickling arbitrary Python objects."""

    def serialize(self, obj, file_path):
        """Serialize the provided object to a CSV file.

        Args:
            obj (any): The object to serialize
            file_path (str): Absolute path to which the object should be
                serialized
        """
        fileio.save_pickle(obj, file_path)
