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
"""Data structures for sparse vector / matrix representations.
"""
# Standard
from typing import Dict

# Third Party
from scipy import sparse
import numpy as np

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from ....core.data_model import DataObjectBase, dataobject
from ....core.toolkit.errors import error_handler
from ...common.data_model import ProducerId
from . import matrix

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(package="watson_core_data_model.nlp")
class Vectorization(DataObjectBase):
    data: Annotated[matrix.SparseMatrix, FieldNumber(1)]
    vocab_to_idx: Annotated[Dict[str, int], FieldNumber(2)]
    producer_id: Annotated[ProducerId, FieldNumber(3)]

    """A Vectorization object, which acts as a lookup table, usually mapping the vocabulary
    to vectors or housing a training result. Note that the `data` property on Vectorization or
    VectorizationPrediction objects is a Scipy CSR matrix for numerical convenience - SparseMatrix
    objects are built only temporarily for conversions in order to tie the data model to the proto
    spec.
    """

    def __init__(
        self,
        data,
        vocab_to_idx,
        keep_dtype=False,
        dtype=np.dtype("float32"),
        producer_id=None,
    ):
        """Initialize a new instance of this class.

        Args:
            data: scipy.sparse.csr_matrix
                2D sparse matrix representing the Vectorization.
            vocab_to_idx: dict
                Mapping from strings to row indices.
            keep_dtype: bool
                Whether to keep the original dtype of the data or not. Default is False.
            dtype: dtype
                The type of the data to convert to. Default is float32.
            producer_id:  ProducerId or None
                The block that produced this vectorization.
        """
        # Require data to be a csr matrix - will only be converted to SparseMatrix when converting
        if isinstance(data, matrix.SparseMatrix):
            error(
                "<NLP78697411E>",
                TypeError(
                    "Although the proto spec uses SparseMatrix objects for serialization, [data] "
                    + "is represented as a scipy CSR matrix on this class for convenience. Pass "
                    + "this initializer the result of the to scipy CSR matrix conversion "
                    + "method on this SparseMatrix instance."
                ),
            )

        error.type_check("<NLP58152341E>", sparse.csr_matrix, data=data)
        error.type_check("<NLP10075393E>", bool, keep_dtype=keep_dtype)
        error.type_check("<NLP97132094E>", np.dtype, dtype=dtype)
        error.type_check(
            "<NLP33722666E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )
        if not isinstance(vocab_to_idx, dict):
            error("<NLP26807643E>", TypeError("`vocab` needs to be dictionary"))
        # Require vocab dict to be a <string, int> pair.
        if not all(
            isinstance(k, str) and isinstance(v, int) for k, v in vocab_to_idx.items()
        ):
            error(
                "<NLP26807638E>",
                TypeError("`vocab` key/value pair is not a <string, int> pair"),
            )
        super().__init__()
        if data.dtype != dtype and not keep_dtype:
            log.warning(
                "<NLP69226577W>",
                "Data type {} will be converted to {}.".format(data.dtype, dtype),
            )
            data = data.astype(dtype)
        self.data = data
        self.vocab_to_idx = vocab_to_idx
        self.producer_id = producer_id

    def fill_proto(self, proto):
        """Override for filling proto on vectorization - this is necessary because we
        need to serialize a scipy CSR matrix. This is accomplished by first converting it to a
        SparseMatrix data model object and then converting that to protobuf.

        Args:
            proto: vectorization_types_pb2.Vectorization
                The vectorization protobuf class object.
        Returns:
            protobuf
                A DataBase object
        """
        # Convert the data scipy CSR matrix to a SparseMatrix so that we can to_proto it
        proto_data = matrix.SparseMatrix.from_scipy_csr_matrix(self.data).to_proto()
        proto.data.CopyFrom(proto_data)
        proto.vocab_to_idx.update(self.vocab_to_idx)
        if self.producer_id is not None:
            proto_producer = self.producer_id.to_proto()
            proto.producer_id.CopyFrom(proto_producer)
        return proto

    @classmethod
    def from_proto(cls, proto):
        """Override for creating an instance of this class from an Vectorization protobuf
        class instance. Note that SparseMatrix is loaded onto the data property as a scipy CSR
        matrix for convenience.

        Args:
            proto: vectorization_types_pb2.Vectorization
                The vectorization prediction protobuf class object.
        Returns:
            EmbeddingPrediction
                An instance of this class.
        """
        # Producer is optional & a data model class - if we have it, we can from_proto it
        prod = ProducerId.from_proto(proto.producer_id) if proto.producer_id else None
        # Pull a sparse matrix out of the proto, then convert that to a scipy csr matrix
        data = matrix.SparseMatrix.from_proto(proto.data).to_scipy_csr_matrix()
        vocab_to_idx = dict(proto.vocab_to_idx)
        return cls(data, vocab_to_idx, keep_dtype=True, producer_id=prod)

    def to_dict(self):
        """Override for dictionary conversion for this object, which is used by .to_json() - we do
        this so that we can stringify the data scipy CSR matrix. Note that we only need to handle
        this for to_json! We get from_json for free because it leverages the other overrides for
        proto conversions.

        Returns:
            dict
                Dictionary representation of an instance of this class.
        """
        return {
            "data": matrix.SparseMatrix.from_scipy_csr_matrix(self.data).to_dict(),
            "vocab_to_idx": self.vocab_to_idx,
            "producer_id": self.producer_id.to_dict() if self.producer_id else None,
        }

    def get_ordered_vocab(self):
        """Retrieve the vocabulary of the model in order, such the the index of each token
        corresponds to its row vector in the data matrix.

        Returns:
            list(str)
                List of ordered tokens.
        """
        sorted_kv_pairs = sorted(self.vocab_to_idx.items(), key=lambda x: x[1])
        return [key for (key, _) in sorted_kv_pairs]


@dataobject(package="watson_core_data_model.nlp")
class VectorizationPrediction(DataObjectBase):
    data: Annotated[matrix.SparseMatrix, FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]

    """A VectorizationPrediction object, which describes the result of running a Vectorization
    model over an input text. Note that the `data` property on Vectorization or
    VectorizationPrediction objects is a Scipy CSR matrix for numerical convenience - SparseMatrix
    objects are built only temporarily for conversions in order to tie the data model to the proto
    spec.
    """

    def __init__(
        self, data, keep_dtype=False, dtype=np.dtype("float32"), producer_id=None
    ):
        """Initialize a new instance of this class.

        Args:
            data: scipy.sparse.csr_matrix
                2D sparse matrix representing the Vectorization Prediction.
            keep_dtype: bool
                Whether to keep the original dtype of the data or not. Default is False.
            dtype: dtype
                The type of the data to convert to. Default is float32.
            producer_id:  ProducerId or None
                The block that produced this vectorization prediction.
        """
        # Require data to be a csr matrix - will only be converted to SparseMatrix when converting
        if isinstance(data, matrix.SparseMatrix):
            error(
                "<NLP51009393E>",
                TypeError(
                    "Although the proto spec uses SparseMatrix objects for serialization, [data] "
                    + "is represented as a scipy CSR matrix on this class for convenience. Pass "
                    + "this initializer the result of the to scipy CSR matrix conversion "
                    + "method on this SparseMatrix instance."
                ),
            )

        error.type_check("<NLP54235393E>", sparse.csr_matrix, data=data)
        error.type_check("<NLP58733683E>", bool, keep_dtype=keep_dtype)
        error.type_check("<NLP14229815E>", np.dtype, dtype=dtype)
        error.type_check(
            "<NLP03311732E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )
        super().__init__()
        if data.dtype != dtype and not keep_dtype:
            log.warning(
                "<NLP14010131W>",
                "Data type {} will be converted to {}.".format(data.dtype, dtype),
            )
            data = data.astype(dtype)
        self.data = data
        self.producer_id = producer_id

    def fill_proto(self, proto):
        """Override for filling proto on vectorization predictions - this is necessary because we
        need to serialize a scipy CSR matrix. This is accomplished by first converting it to a
        SparseMatrix data model object and then converting that to protobuf.

        Args:
            proto: vectorization_types_pb2.VectorizationPrediction
                The vectorization prediction protobuf class object.
        Returns:
            protobuf
                A DataBase object
        """
        # Convert the data scipy CSR matrix to a SparseMatrix so that we can to_proto it
        proto_data = matrix.SparseMatrix.from_scipy_csr_matrix(self.data).to_proto()
        proto.data.CopyFrom(proto_data)
        if self.producer_id is not None:
            proto_producer = self.producer_id.to_proto()
            proto.producer_id.CopyFrom(proto_producer)
        return proto

    @classmethod
    def from_proto(cls, proto):
        """Override for creating an instance of this class from an VectorizationPrediction protobuf
        class instance. Note that SparseMatrix is loaded onto the data property as a scipy CSR
        matrix for convenience.

        Args:
            proto: vectorization_types_pb2.VectorizationPrediction
                The vectorization prediction protobuf class object.
        Returns:
            EmbeddingPrediction
                An instance of this class.
        """
        # Producer is optional & a data model class - if we have it, we can from_proto it
        prod = ProducerId.from_proto(proto.producer_id) if proto.producer_id else None
        # Pull a sparse matrix out of the proto, then convert that to a scipy csr matrix
        data = matrix.SparseMatrix.from_proto(proto.data).to_scipy_csr_matrix()
        return cls(data, keep_dtype=True, producer_id=prod)

    def to_dict(self):
        """Override for dictionary conversion for this object, which is used by .to_json() - we do
        this so that we can stringify the data scipy CSR matrix. Note that we only need to handle
        this for to_json! We get from_json for free because it leverages the other overrides for
        proto conversions.

        Returns:
            dict
                Dictionary representation of an instance of this class.
        """
        return {
            "data": matrix.SparseMatrix.from_scipy_csr_matrix(self.data).to_dict(),
            "producer_id": self.producer_id.to_dict() if self.producer_id else None,
        }
