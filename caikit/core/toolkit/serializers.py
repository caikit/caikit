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
from typing import List
import abc

# Local
from . import fileio

# from caikit.core.signature_parsing.module_signature import CaikitMethodSignature



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


class _DocstringSerializer:
    def __init__(self, method):
        self._method = method
        self._docstring = getattr(self._method.module, self._method.method_name).__doc__

    def _line(self, line="", indent_to_remove: str = "", extra_indent: str = "") -> str:
        return f"{extra_indent}{line.replace(indent_to_remove, '', 1)}"

    def _get_indent(self, docstring_lines: List[str]) -> str:
        """Return a whitespace string representing the indent level of the docstring

        Args:
            docstring_lines: docstring split up into lines
        """

        # Find first non-blank line that isn't the first line
        for line in docstring_lines[1:]:
            if len(line.strip()) > 0:
                # return the indent
                # yolo swaggins this doesn't work for tabs
                return " " * (len(line) - len(line.lstrip()))
        return ""

    def _strip_last_blank_line(self, doc_lines: List[str]) -> List[str]:
        list_copy = doc_lines[:]
        while len(list_copy[-1].strip()) == 0:
            list_copy.pop()

        return list_copy

    def to_dot_proto_lines(self, extra_indent: str = "") -> List[str]:
        # log.debug4("Serializing docstring for method %s", self._method)
        lines = []
        if not self._docstring:
            return ""
        doc_lines: List[str] = self._docstring.split("\n")
        doc_lines = self._strip_last_blank_line(doc_lines)

        docstring_indent = self._get_indent(doc_lines)

        for doc_line in doc_lines:
            lines.append(
                self._line(
                    doc_line,
                    indent_to_remove=docstring_indent,
                    extra_indent=extra_indent,
                )
            )

        return lines
