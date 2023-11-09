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
"""Data structures for embedding vector representations
"""
# Standard
from dataclasses import dataclass, field
from typing import Any, List, Union
import json

# Third Party
from google.protobuf import json_format
import numpy as np

# First Party
import alog

# Local
from caikit.core import DataObjectBase, dataobject
from caikit.core.data_model import PACKAGE_COMMON
from caikit.core.exceptions import error_handler

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(PACKAGE_COMMON)
@dataclass
class PyFloatSequence(DataObjectBase):
    values: List[float] = field(default_factory=list)


@dataobject(PACKAGE_COMMON)
@dataclass
class NpFloat32Sequence(DataObjectBase):
    values: List[np.float32]

    @classmethod
    def from_proto(cls, proto):
        values = np.asarray(proto.values, dtype=np.float32)
        return cls(values)


@dataobject(PACKAGE_COMMON)
@dataclass
class NpFloat64Sequence(DataObjectBase):
    values: List[np.float64]

    @classmethod
    def from_proto(cls, proto):
        values = np.asarray(proto.values, dtype=np.float64)
        return cls(values)


@dataobject(PACKAGE_COMMON)
@dataclass
class Vector1D(DataObjectBase):
    """Data representation for a 1 dimension vector of float-type data."""

    data: Union[
        PyFloatSequence,
        NpFloat32Sequence,
        NpFloat64Sequence,
    ]

    def __post_init__(self):
        error.value_check(
            "<NLP92989048E>",
            hasattr(self.data, "values"),
            ValueError("Vector1D requires a float sequence data object with values."),
        )

    @classmethod
    def from_vector(cls, vector):
        dtype = getattr(vector, "dtype", False)
        if dtype is None:
            data = PyFloatSequence(vector)
        elif dtype == np.float32:
            data = NpFloat32Sequence(vector)
        elif dtype == np.float64:
            data = NpFloat64Sequence(vector)
        else:
            data = PyFloatSequence(vector)
        return cls(data=data)

    @classmethod
    def from_json(cls, json_str: Union[dict[str, Any], str]) -> "Vector1D":
        """JSON does not have different float types. Move data into data_pyfloatsequence"""

        json_obj = json.loads(json_str) if isinstance(json_str, str) else json_str
        data = json_obj.pop("data")
        if data is not None:
            json_obj["data_pyfloatsequence"] = data

        json_str = json.dumps(json_obj)
        try:
            # Parse given JSON into google.protobufs.pyext.cpp_message.GeneratedProtocolMessageType
            parsed_proto = json_format.Parse(
                json_str, cls.get_proto_class()(), ignore_unknown_fields=False
            )

            # Use from_proto to return the DataBase object from the parsed proto
            return cls.from_proto(parsed_proto)

        except json_format.ParseError as ex:
            error("<NLP39795399E>", ValueError(ex))

    def to_dict(self) -> dict:
        """to_dict is needed to make things serializable"""
        values = self.data.values if self.data.values is not None else []
        return {
            "data": {
                # coerce numpy.ndarray and numpy.float32 into JSON serializable list of floats
                "values": values.tolist()
                if isinstance(values, np.ndarray)
                else values
            }
        }

    @classmethod
    def from_proto(cls, proto):
        """Wrap the data in an appropriate float sequence, wrapped by this class"""
        woo = proto.WhichOneof("data")
        if woo is None:
            return cls(PyFloatSequence())

        woo_data = getattr(proto, woo)
        if woo == "data_npfloat64sequence":
            ret = cls(NpFloat64Sequence.from_proto(woo_data))
        elif woo == "data_npfloat32sequence":
            ret = cls(NpFloat32Sequence.from_proto(woo_data))
        else:
            ret = cls(PyFloatSequence.from_proto(woo_data))
        return ret

    def fill_proto(self, proto):
        """Fill in the data in an appropriate data_<float type sequence>"""
        values = self.data.values
        if values is not None and len(values) > 0:
            sample = values[0]
            error.type_check(
                "<NLP47515960E>", float, np.float32, np.float64, sample=sample
            )
            if isinstance(sample, np.float64):
                proto.data_npfloat64sequence.values.extend(values)
            elif isinstance(sample, np.float32):
                proto.data_npfloat32sequence.values.extend(values)
            else:
                proto.data_pyfloatsequence.values.extend(values)

        return proto


@dataobject(PACKAGE_COMMON)
class ListOfVector1D(DataObjectBase):
    """Data representation for an embedding matrix holding 2D vectors"""

    results: List[Vector1D]

    def __post_init__(self):
        error.type_check("<NLP94336739E>", list, results=self.results)
        error.type_check_all("<NLP94783841E>", Vector1D, results=self.results)

    @classmethod
    def from_json(cls, json_str: Union[dict[str, Any], str]) -> "ListOfVector1D":
        """Fill in the vector data in an appropriate data_<float type sequence>"""

        json_obj = json.loads(json_str) if isinstance(json_str, str) else json_str
        for v in json_obj["results"]:
            data = v.pop("data")
            if data is not None:
                v["data_pyfloatsequence"] = data
        json_str = json.dumps(json_obj)
        try:
            # Parse given JSON into google.protobufs.pyext.cpp_message.GeneratedProtocolMessageType
            parsed_proto = json_format.Parse(
                json_str, cls.get_proto_class()(), ignore_unknown_fields=False
            )

            # Use from_proto to return the DataBase object from the parsed proto
            return cls.from_proto(parsed_proto)

        except json_format.ParseError as ex:
            error("<NLP85443094E>", ValueError(ex))
